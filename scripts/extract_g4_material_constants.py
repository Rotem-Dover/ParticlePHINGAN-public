"""One-shot extractor for Fe/Be model constants from the local GEANT4 11.2.0
source. Prints ready-to-paste Python. Refuses to print anything unless the
same parse reproduces the Al/Si values already committed in
physics/g4_init_tables/models/ (regression gate against mis-parsing).

Run from the repo root:  PYTHONPATH=src .venv-python scripts/extract_g4_material_constants.py
"""
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from physics.g4_init_tables.models import _pstar_tables as P           # noqa: E402
from physics.g4_init_tables.models.bragg import _ICRU49                # noqa: E402
from physics.g4_init_tables.models.shell_correction import (           # noqa: E402
    ATOMIC_SHELL_ELECTRONS,
)

G4 = Path.home() / "Documents/geant4/geant4-11.2.0/source"
STD = G4 / "processes/electromagnetic/standard"


def pstar() -> None:
    hh = (STD / "include/G4NISTStoppingData.hh").read_text()
    names = re.findall(r'"(G4_[^"]*)"', hh.split("nameNIST")[1].split("};")[0])
    cc = (STD / "src/G4PSTARStopping.cc").read_text()

    def arr(idx: int) -> list[float]:
        m = re.search(rf"static const G4float e{idx}\[60\] = \{{(.*?)\}}", cc, re.S)
        return [float(v.strip().rstrip("f")) for v in m.group(1).split(",")]

    al = torch.tensor(arr(names.index("G4_Al")), dtype=torch.float64)
    assert torch.equal(al, P.DEDX_Al_MeV_cm2_per_g), "PSTAR Al regression FAILED"
    if "G4_Si" in names:
        si = torch.tensor(arr(names.index("G4_Si")), dtype=torch.float64)
        assert torch.equal(si, P.DEDX_Si_MeV_cm2_per_g), "PSTAR Si regression FAILED"
    for g4, tag in (("G4_Be", "Be"), ("G4_Fe", "Fe")):
        vals = arr(names.index(g4))
        print(f"DEDX_{tag}_MeV_cm2_per_g = torch.tensor({vals}, dtype=torch.float64)")


def sternheimer() -> None:
    cc = (G4 / "materials/src/G4DensityEffectData.cc").read_text()

    def row(tag: str) -> list[float]:
        m = re.search(
            rf'G4double M\d+\[NDENSARRAY\] = \{{\s*([^}}]+)\}};\s*\n\s*AddMaterial\(M\d+, "{tag}"\)',
            cc)
        return [float(x) for x in m.group(1).replace("\n", " ").split(",")]

    # layout: [Eplasma, adj, C, x0, x1, a, m, delta0, err, I]
    al = row("G4_Al")
    assert al[2:8] == [4.2395, 0.1708, 3.0127, 0.08024, 3.6345, 0.12], "Sternheimer Al regression FAILED"
    for tag, name in (("G4_Fe", "Iron"), ("G4_Be", "Beryllium")):
        r = row(tag)
        print(f'    "{name}": SternheimerParams(x0={r[3]}, x1={r[4]}, m={r[6]}, '
              f'a={r[5]}, C_={r[2]}, delta0={r[7]}),')


def icru49() -> None:
    cc = (STD / "src/G4BraggModel.cc").read_text()
    m = re.search(r"static const G4float a\[92\]\[5\] = \{(.*?)\n  \};", cc, re.S)
    # Some rows have a commented-out alternative on their own "// {...}"
    # line (Ag Ziegler77, Pt Ziegler77) which also matches a naive brace
    # scan -- drop full comment-only lines before extracting rows.
    body = "\n".join(
        line for line in m.group(1).splitlines() if not line.strip().startswith("//")
    )
    rows = re.findall(r"\{([^}]+)\}", body)
    tab = [[float(v.strip().rstrip("f")) for v in r.split(",")] for r in rows]
    assert len(tab) == 92
    assert tuple(tab[12]) == _ICRU49[13], "ICRU49 Z=13 regression FAILED"
    assert tuple(tab[13]) == _ICRU49[14], "ICRU49 Z=14 regression FAILED"
    for z, label in ((4, "Beryllium"), (26, "Iron")):
        print(f"    {z}: {tuple(tab[z - 1])},   # {label}")


def shells() -> None:
    cc = (G4 / "materials/src/G4AtomicShells.cc").read_text()

    def ints(after: str) -> list[int]:
        block = cc.split(after)[1].split("};")[0]
        # Strip `// ...` line comments first -- they contain digits of their
        # own (element names' Z ranges, e.g. "//   1 - 10"), which would
        # otherwise contaminate the flat int list.
        block = re.sub(r"//[^\n]*", "", block)
        return [int(x) for x in re.findall(r"-?\d+", block)]

    n_shells = ints("fNumberOfShells[105] =")
    electrons = ints("fNumberOfElectrons[1650] =")

    def shells_for(z: int) -> list[int]:
        off = sum(n_shells[:z])
        return electrons[off:off + n_shells[z]]

    assert shells_for(13) == ATOMIC_SHELL_ELECTRONS[13], "shells Z=13 regression FAILED"
    assert shells_for(14) == ATOMIC_SHELL_ELECTRONS[14], "shells Z=14 regression FAILED"
    for z, label in ((4, "Beryllium"), (26, "Iron")):
        print(f"    {z}: {shells_for(z)},   # {label}")


if __name__ == "__main__":
    print("# --- paste into _pstar_tables.py ---"); pstar()
    print("# --- paste into bethe_bloch._STERNHEIMER ---"); sternheimer()
    print("# --- paste into bragg._ICRU49 ---"); icru49()
    print("# --- paste into shell_correction.ATOMIC_SHELL_ELECTRONS ---"); shells()
