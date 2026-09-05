"""
Torch port of G4PhysicsVector / G4PhysicsLogVector with cubic-spline
interpolation. Matches the geometry of binVector/dataVector/secDerivative and
the Interpolation kernel from
geant4-11.2.0/source/global/management/src/G4PhysicsVector.cc.

Second-derivative method matches G4's runtime default: `FillSecondDerivatives`
with `G4SplineType::Base` -> `ComputeSecDerivative1` (Not-a-knot endpoint
conditions, tridiagonal solve), per `G4PhysicsVector::FillSecondDerivatives`
(lines 207-271 of G4PhysicsVector.cc) which dispatches `Base` to
`ComputeSecDerivative1` (lines 292-354). All tables built by
`G4LossTableBuilder::BuildRangeTable` etc. call `FillSecondDerivatives()`
with no args, so this is the spline G4 actually evaluates at runtime.
"""
from __future__ import annotations

import torch

from physics.g4_init_tables._g4math import g4_log_jit, g4_log_scalar


# @torch.jit.script is deliberate: letting inductor inline this kernel
# introduces 1-3 ULP codegen drift on the G4 bit-contract with no
# corresponding speedup (CUDA graphs already amortize the graph break), so
# the scripted, bit-exact form is kept and it graph-breaks out of the
# compiled phases.
@torch.jit.script
def g4pv_eval(
    bin: torch.Tensor,        # float64 [N] grid nodes (energy or range)
    data: torch.Tensor,       # float64 [N] table values
    sec_deriv: torch.Tensor,  # float64 [N] spline second derivatives
    log_emin: float,          # G4Log(bin[0]) for log grids; 0.0 for free
    inv_dbin_log: float,      # (N-1)/G4Log(bin[-1]/bin[0]) for log; 0.0 free
    log_grid: bool,
    use_spline: bool,
    e: torch.Tensor,
) -> torch.Tensor:
    """Scripted port of G4PhysicsVector::Value (linear + Base-spline cubic).

    Branchless (no .any() / data-dependent control flow) so it never forces
    a GPU->CPU sync inside the stepping hot loop. Out-of-range queries clamp
    to the edge values, matching G4PhysicsVector::Value semantics.
    Evaluates in float64; callers cast to their state dtype.
    """
    e64 = e.to(torch.float64)
    edge_min = bin[0]
    edge_max = bin[-1]
    idxmax = bin.shape[0] - 2

    e_in = torch.clamp(e64, edge_min, edge_max)
    if log_grid:
        # G4Log (not torch.log): bin selection must match G4's
        # ComputeLogVectorBin to the last ULP (see class docstring).
        idx = ((g4_log_jit(e_in) - log_emin) * inv_dbin_log).to(torch.long)
    else:
        idx = torch.bucketize(e_in.contiguous(), bin, right=False) - 1
    idx = idx.clamp(0, idxmax)

    x1 = bin[idx]
    dl = bin[idx + 1] - x1
    y1 = data[idx]
    dy = data[idx + 1] - y1
    b = (e_in - x1) / dl
    res = y1 + b * dy
    if use_spline:
        c0 = (2.0 - b) * sec_deriv[idx]
        c1 = (1.0 + b) * sec_deriv[idx + 1]
        res = res + (b * (b - 1.0)) * (c0 + c1) * (dl * dl * (1.0 / 6.0))

    res = torch.where(e64 <= edge_min, data[0], res)
    res = torch.where(e64 >= edge_max, data[-1], res)
    return res


@torch.jit.script
def g4pv_inverse_range_eval(
    bin: torch.Tensor, data: torch.Tensor, sec_deriv: torch.Tensor,
    min_kin_energy: float, r: torch.Tensor,
) -> torch.Tensor:
    """Inverse-range lookup with G4's below-first-node rule.

    Port of G4VEnergyLossProcess::ScaledKinEnergyForLoss (verified against
    geant4 v11.2.0, G4VEnergyLossProcess.hh — inline definition):

        G4double rmin = v->Energy(0);
        if(r >= rmin)    { e = v->Value(r, idxInverseRange); }
        else if(r > 0.0) { G4double x = r/rmin; e = minKinEnergy*x*x; }
        else             { e = 0.0; }

    `min_kin_energy` is G4's `minKinEnergy` (= G4EmParameters::MinKinEnergy,
    default 100 eV) — the same member used by the sqrt range-scaling in
    GetScaledRangeForScaledEnergy, so callers pass the same
    GeantConstants.min_kin_energy for both. Note the reference simulation's
    PhysListEmStandard sets MinEnergy=10 eV; both rules only fire in the
    final ~nm of a track, far below where the emulation operates, and the
    100 eV constant is the one already validated against the G4
    fluctuation logs.
    """
    r64 = r.to(torch.float64)
    rmin = bin[0]
    e = g4pv_eval(bin, data, sec_deriv, 0.0, 0.0, False, True, r64)
    ratio = r64 / rmin
    quad = min_kin_energy * ratio * ratio
    e = torch.where(r64 < rmin, quad, e)
    return torch.where(r64 <= 0.0, torch.zeros_like(e), e)


class G4PhysicsVector:
    def __init__(
        self,
        e_grid: torch.Tensor,
        values: torch.Tensor,
        log_grid: bool = True,
        use_spline: bool = True,
    ):
        if e_grid.shape != values.shape or e_grid.ndim != 1:
            raise ValueError("e_grid and values must be 1-D and identically shaped")
        if e_grid.numel() < 2:
            raise ValueError("need at least 2 nodes")

        self.bin = e_grid.to(torch.float64).contiguous()
        self.data = values.to(torch.float64).contiguous()
        self.log_grid = log_grid
        self.use_spline = use_spline and self.bin.numel() >= 5
        self.idxmax = self.bin.numel() - 2
        self.edge_min = float(self.bin[0])
        self.edge_max = float(self.bin[-1])

        if log_grid:
            # Use G4Log (not libm) so `log_emin` / `inv_dbin_log` match
            # G4's compiled `logemin = G4Log(edgeMin); invdBin = (idxmax+1)/
            # G4Log(edgeMax/edgeMin)` bit-for-bit. With libm `math.log`,
            # near-threshold queries (e.g. lambda_ion just above E_thr) can
            # cross an ULP-level bin boundary at a different energy than G4,
            # which shifts the spline's cubic correction in that bin.
            self.log_emin = g4_log_scalar(self.edge_min)
            log_emax = g4_log_scalar(self.edge_max)
            # G4PhysicsLogVector::Initialise uses invdBin = (idxmax + 1) /
            # log(edgeMax/edgeMin), not idxmax/log(...). Using idxmax instead
            # of idxmax+1 pushes higher-bin queries one bin to the left,
            # which changes which spline segment they are evaluated on.
            self.inv_dbin_log = (self.idxmax + 1) / (log_emax - self.log_emin)
        else:
            self.log_emin = 0.0
            self.inv_dbin_log = 0.0

        if self.use_spline:
            self.sec_deriv = self._compute_sec_derivative1()
        else:
            self.sec_deriv = torch.zeros_like(self.bin)

    @classmethod
    def from_log_grid(cls, e_min: float, e_max: float, n_nodes: int, values: torch.Tensor, **kw) -> "G4PhysicsVector":
        from physics.g4_init_tables.grid import log_energy_grid
        e_grid = log_energy_grid(e_min, e_max, n_nodes)
        return cls(e_grid, values, log_grid=True, **kw)

    def _compute_sec_derivative1(self) -> torch.Tensor:
        """Port of G4PhysicsVector::ComputeSecDerivative1 (Not-a-knot endpoints).

        Mirrors the tridiagonal solve at G4PhysicsVector.cc lines 292-354 verbatim;
        kept as a sequential scalar loop because the recurrence is forward-
        dependent (each step uses the previous step's `secDerivative` and `u`).
        n_nodes is small (~10² for G4 init tables), so this is not the hot path.
        """
        bv = self.bin
        dv = self.data
        N = bv.numel()
        n = N - 1
        sec = torch.zeros(N, dtype=torch.float64)
        u = torch.zeros(n, dtype=torch.float64)

        u[1] = ((dv[2] - dv[1]) / (bv[2] - bv[1]) -
                (dv[1] - dv[0]) / (bv[1] - bv[0]))
        u[1] = 6.0 * u[1] * (bv[2] - bv[1]) / \
               ((bv[2] - bv[0]) * (bv[2] - bv[0]))

        sec[1] = (2.0 * bv[1] - bv[0] - bv[2]) / \
                 (2.0 * bv[2] - bv[0] - bv[1])

        for i in range(2, n - 1):
            sig = (bv[i] - bv[i - 1]) / (bv[i + 1] - bv[i - 1])
            p = sig * sec[i - 1] + 2.0
            sec[i] = (sig - 1.0) / p
            u[i] = ((dv[i + 1] - dv[i]) / (bv[i + 1] - bv[i]) -
                    (dv[i] - dv[i - 1]) / (bv[i] - bv[i - 1]))
            u[i] = (6.0 * u[i] / (bv[i + 1] - bv[i - 1])) - sig * u[i - 1] / p

        sig = (bv[n - 1] - bv[n - 2]) / (bv[n] - bv[n - 2])
        p = sig * sec[n - 3] + 2.0
        u[n - 1] = ((dv[n] - dv[n - 1]) / (bv[n] - bv[n - 1]) -
                    (dv[n - 1] - dv[n - 2]) / (bv[n - 1] - bv[n - 2]))
        u[n - 1] = 6.0 * sig * u[n - 1] / (bv[n] - bv[n - 2]) - \
                   (2.0 * sig - 1.0) * u[n - 2] / p

        p = (1.0 + sig) + (2.0 * sig - 1.0) * sec[n - 2]
        sec[n - 1] = u[n - 1] / p

        # back-substitution
        for k in range(n - 2, 1, -1):
            sec[k] = sec[k] * (sec[k + 1] -
                               u[k] * (bv[k + 1] - bv[k - 1]) /
                               (bv[k + 1] - bv[k]))

        sec[n] = (sec[n - 1] - (1.0 - sig) * sec[n - 2]) / sig
        sig = 1.0 - ((bv[2] - bv[1]) / (bv[2] - bv[0]))
        sec[1] = sec[1] * (sec[2] - u[1] / (1.0 - sig))
        sec[0] = (sec[1] - sig * sec[2]) / (1.0 - sig)

        return sec

    def __call__(self, e: torch.Tensor) -> torch.Tensor:
        return g4pv_eval(self.bin, self.data, self.sec_deriv,
                         self.log_emin, self.inv_dbin_log,
                         self.log_grid, self.use_spline, e)

    def to(self, device=None, dtype=None) -> "G4PhysicsVector":
        if device is not None:
            self.bin = self.bin.to(device)
            self.data = self.data.to(device)
            self.sec_deriv = self.sec_deriv.to(device)
        if dtype is not None:
            self.bin = self.bin.to(dtype)
            self.data = self.data.to(dtype)
            self.sec_deriv = self.sec_deriv.to(dtype)
        return self
