"""
Draw the network-architecture figure for the paper's appendix.

The figure is derived, not transcribed: the script instantiates the two
published generators (`ContinuousGenerator`, `SecondaryGenerator`) and the two
critics they were trained against, walks their `nn.Sequential` stacks, and
draws one row per `Linear` layer with the normalisation and activation that
follow it. Every width, norm and terminal therefore comes from the checkpoint
contract in `common.config`; changing a constant there changes the figure.

Usage (from the repository root, with the project's virtualenv active):

    PYTHONPATH=src python scripts/draw_architecture.py [--out DIR]

Writes `architecture.pdf` and `architecture.png` to DIR (default:
`artifacts/benchmark_outputs/`).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from torch import nn

from common.config import (N_NOISE_CNT, N_NOISE_SEC, N_CONTINUOUS_FEATURES,
                           N_SECONDARY_FEATURES)
from common.paths import BENCHMARKS_ROOT
from physics.g4h_ionisation.generators.continuous_generator.neural_net import ContinuousGenerator
from physics.g4h_ionisation.generators.secondary_generator.neural_net import SecondaryGenerator
from physics.g4h_ionisation.generators.train_networks.continuous_generator.critic import Critic as ContinuousCritic
from physics.g4h_ionisation.generators.train_networks.secondary_generator.critic import Critic as SecondaryCritic


# ----------------------------------------------------------------------------
# 1. Turn a module into rows: (Linear, [norm], [activation], [terminal])
# ----------------------------------------------------------------------------

@dataclass
class Row:
    """One drawn row: a Linear layer and whatever follows it before the next."""
    n_in: int
    n_out: int
    tail: list[str] = field(default_factory=list)   # e.g. ["LayerNorm", "ReLU"]


_TERMINAL_LABELS = {
    "HardtanhReflectTop": "clamp [0, 1], reflect top",
    "HardtanhOpenTop":    "clamp [0, 1]",
    "Hardtanh":           "clamp [0, 1]",
    "StretchedSigmoid":   "stretched sigmoid",
    "Sigmoid":            "sigmoid",
}


def _leaf_label(m: nn.Module) -> str:
    name = type(m).__name__
    if isinstance(m, nn.LeakyReLU):
        return f"LeakyReLU {m.negative_slope:g}"
    if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.InstanceNorm1d)):
        return name.replace("1d", "")
    return _TERMINAL_LABELS.get(name, name)


def rows_of(seq: nn.Module) -> list[Row]:
    """Greedy grouping: a row starts at each Linear and absorbs what follows."""
    rows: list[Row] = []
    for leaf in (m for m in seq.modules() if not list(m.children())):
        if isinstance(leaf, nn.Linear):
            rows.append(Row(leaf.in_features, leaf.out_features))
        elif isinstance(leaf, nn.Flatten):
            continue
        elif rows:
            rows[-1].tail.append(_leaf_label(leaf))
    return rows


@dataclass
class NetSpec:
    title: str
    cond_label: str            # what is embedded
    side_label: str            # what is concatenated after the embedding
    side_width: int
    out_label: str
    embed: list[Row]
    trunk: list[Row]


def generator_spec(net, title, cond_label, out_label, n_noise) -> NetSpec:
    return NetSpec(title, cond_label,
                   side_label=fr"latent $z \sim \mathcal{{N}}(0, I_{{{n_noise}}})$",
                   side_width=n_noise, out_label=out_label,
                   embed=rows_of(net.embed), trunk=rows_of(net.model))


def critic_spec(net, title, cond_label, sample_label, n_feat) -> NetSpec:
    return NetSpec(title, cond_label,
                   side_label=f"candidate {sample_label}\n(truth or generated)",
                   side_width=n_feat, out_label="critic score",
                   embed=rows_of(net.embed), trunk=rows_of(net.model))


# ----------------------------------------------------------------------------
# 2. Drawing
# ----------------------------------------------------------------------------

PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

# Muted, print-safe palette. One hue per kind of operation.
C_LINEAR = "#cfe0f3"      # weights
C_NORM   = "#e6e6e6"      # normalisation
C_ACT    = "#fbdcb5"      # nonlinearity
C_TERM   = "#cde9cc"      # output map
C_EDGE   = "#333333"
C_IO     = "#ffffff"

ROW_H   = 0.62            # height of a layer row (data units)
ROW_GAP = 0.22
COL_W   = 5.4             # width of a layer row
PILL_H  = 0.5


def _box(ax, x, y, w, h, fc, text, *, fs=None, bold=False, ec=C_EDGE, lw=0.8, pad=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={pad}",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal")


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=9, lw=0.8, color=C_EDGE,
                                 shrinkA=0, shrinkB=0))


def _row(ax, x, y, row: Row):
    """A Linear row: the weight segment, then one segment per trailing op."""
    parts = [(f"Linear  {row.n_in} → {row.n_out}", C_LINEAR)]
    for t in row.tail:
        if t.startswith(("clamp", "sigmoid", "stretched")):
            parts.append((t, C_TERM))
        elif "Norm" in t:
            parts.append((t, C_NORM))
        else:
            parts.append((t, C_ACT))
    # The weight segment gets the lion's share; the others split the rest.
    weights = [1.0] + [0.72 if "Norm" in p[0]
                       else 1.05 if "Leaky" in p[0]
                       else 1.6 if p[1] == C_TERM
                       else 0.72
                       for p in parts[1:]]
    total = sum(weights)
    cx = x
    for (label, color), wgt in zip(parts, weights):
        w = COL_W * wgt / total
        _box(ax, cx, y, w, ROW_H, color, label, pad=0.06, fs=7.8)
        cx += w


def draw_net(ax, spec: NetSpec, x0: float, y_top: float) -> float:
    """Draw one network with its top-left at (x0, y_top). Returns bottom y."""
    cx = x0 + COL_W / 2
    y = y_top

    ax.text(cx, y, spec.title, ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    y -= 0.15

    # conditioning input
    y -= PILL_H
    _box(ax, x0 + COL_W * 0.2, y, COL_W * 0.6, PILL_H, C_IO, spec.cond_label, pad=0.2)
    y_prev = y

    # embedding rows
    for row in spec.embed:
        y -= ROW_GAP + ROW_H
        _arrow(ax, cx, y_prev, cx, y + ROW_H)
        _row(ax, x0, y, row)
        y_prev = y
    n_embed = spec.embed[-1].n_out
    y_emb_top = y_top - 0.15 - PILL_H - ROW_GAP
    ax.annotate("", xy=(x0 - 0.08, y_prev), xytext=(x0 - 0.08, y_emb_top),
                arrowprops=dict(arrowstyle="-", lw=0.8, color="#555555", shrinkA=0, shrinkB=0))
    ax.text(x0 - 0.16, (y_prev + y_emb_top) / 2, "embedding", rotation=90,
            ha="right", va="center", fontsize=7.5, style="italic", color="#555555")

    # concatenation with the side input
    y -= ROW_GAP + ROW_H
    _arrow(ax, cx, y_prev, cx, y + ROW_H)
    n_cat = n_embed + spec.side_width
    _box(ax, x0 + COL_W * 0.15, y, COL_W * 0.70, ROW_H, C_IO,
         f"concat  [{n_embed} ‖ {spec.side_width}]  →  {n_cat}", pad=0.06)
    # side pill to the right of the column
    sx = x0 + COL_W + 0.35
    _box(ax, sx, y + (ROW_H - PILL_H * 1.5) / 2, COL_W * 0.64, PILL_H * 1.5, C_IO,
         spec.side_label, pad=0.2)
    _arrow(ax, sx, y + ROW_H / 2, x0 + COL_W * 0.85, y + ROW_H / 2)
    y_prev = y

    # trunk rows
    for row in spec.trunk:
        y -= ROW_GAP + ROW_H
        _arrow(ax, cx, y_prev, cx, y + ROW_H)
        _row(ax, x0, y, row)
        y_prev = y

    # output
    y -= ROW_GAP + PILL_H
    _arrow(ax, cx, y_prev, cx, y + PILL_H)
    _box(ax, x0 + COL_W * 0.2, y, COL_W * 0.6, PILL_H, C_IO, spec.out_label, pad=0.2, bold=True)
    return y


def build_specs() -> list[list[NetSpec]]:
    g_cnt = ContinuousGenerator().eval()
    g_sec = SecondaryGenerator().eval()
    d_cnt = ContinuousCritic().eval()
    d_sec = SecondaryCritic().eval()
    return [
        [generator_spec(g_cnt, "Continuous-loss generator", "$(E, L)$",
                        r"$\Delta E_{\mathrm{cnt}}$", N_NOISE_CNT),
         critic_spec(d_cnt, "Continuous-loss critic", "$(E, L)$",
                     r"$\Delta E_{\mathrm{cnt}}$", N_CONTINUOUS_FEATURES)],
        [generator_spec(g_sec, "Secondary generator", "$E$",
                        r"$\Delta E_{\mathrm{sec}}$", N_NOISE_SEC),
         critic_spec(d_sec, "Secondary critic", "$E$",
                     r"$\Delta E_{\mathrm{sec}}$", N_SECONDARY_FEATURES)],
    ]


def draw_figure() -> plt.Figure:
    specs = build_specs()
    col_pitch = COL_W + 0.35 + COL_W * 0.64 + 0.6      # column + side pill + margin
    fig, ax = plt.subplots(figsize=(8.6, 9.6))
    ax.set_axis_off()

    y_top = 0.0
    for stage_row in specs:
        bottoms = []
        for j, spec in enumerate(stage_row):
            bottoms.append(draw_net(ax, spec, j * col_pitch, y_top))
        y_top = min(bottoms) - 1.0
    y_top += 0.6

    ax.set_xlim(-0.6, 2 * col_pitch - 0.4)
    ax.set_ylim(y_top - 0.1, 0.5)
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.2)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=BENCHMARKS_ROOT,
                   help="output directory (default: artifacts/benchmark_outputs/)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(PAPER_RC):
        fig = draw_figure()
        for ext in ("pdf", "png"):
            path = args.out / f"architecture.{ext}"
            fig.savefig(path, bbox_inches="tight", dpi=200)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
