"""Report figures (static PNGs for the README).

Colours are the first categorical slots of a colour-blind-validated palette
(blue, orange, aqua), always paired with a marker shape or label so no
category is told apart by colour alone. One y-axis per panel.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import clutter  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    })


# --------------------------------------------------------------------------- #
# Synthetic checks
# --------------------------------------------------------------------------- #
def plot_far(df: pd.DataFrame):
    """Measured vs design false-alarm rate, one line per detector/clutter case."""
    style()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    lim = [df.design_pfa.min() / 3, max(df.measured_pfa.max(), df.design_pfa.max()) * 3]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--", label="measured = design")
    for (case, g), color, mk in zip(df.groupby("case", sort=False), [AQUA, BLUE, ORANGE, VIOLET], "osD^"):
        g = g.sort_values("design_pfa")
        y = np.maximum(g.measured_pfa, lim[0])
        ax.plot(g.design_pfa, y, color=color, marker=mk, ms=6, label=case)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="design false-alarm probability",
           ylabel="measured false-alarm rate", title="CFAR false-alarm rate, synthetic clutter")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_pd_scr(scr_db, measured, n_per_level, theory_scr, theory_pd, pfa):
    style()
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(theory_scr, theory_pd, color=BLUE, label=f"theory (CA-CFAR, Pfa {pfa:g})")
    from .validation import wilson
    lo, hi = wilson(np.asarray(measured) * n_per_level, np.full(len(measured), n_per_level))
    ax.errorbar(scr_db, measured, yerr=np.clip([measured - lo, hi - measured], 0, None), fmt="o", color=ORANGE, ms=6,
                capsize=2, lw=1.2, label=f"measured ({n_per_level} targets per level, 95 % CI)")
    ax.set(xlabel="signal-to-clutter ratio of a one-pixel target (dB)", ylabel="detection probability",
           ylim=(-0.03, 1.03), title="Detection probability vs SCR, synthetic")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_length_synthetic(df: pd.DataFrame):
    """Estimated vs true length for raw and blur-corrected estimators, one panel per case."""
    style()
    cases = list(dict.fromkeys(df.case))
    fig, axs = plt.subplots(1, len(cases), figsize=(3.4 * len(cases), 3.4), squeeze=False)
    for ax, case in zip(axs[0], cases):
        g = df[df.case == case]
        top = max(g.true_m.max(), g.length_core_m.max()) * 1.1
        ax.plot([0, top], [0, top], color=INK2, lw=1, ls="--")
        ax.scatter(g.true_m, g.length_core_m, s=16, color=ORANGE, marker="s", label="core extent")
        ax.scatter(g.true_m, g.length_psf_m, s=16, color=BLUE, marker="o", label="blur-corrected")
        ax.set(xlim=(0, top), ylim=(0, top), xlabel="true length (m)", title=case)
    axs[0, 0].set_ylabel("estimated length (m)")
    axs[0, 0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Scene overview
# --------------------------------------------------------------------------- #
def plot_scene(scene, val, title: str, water=None, vmin=-25.0, vmax=0.0, figsize=(10, 9),
               ref: str = "AIS vessel"):
    """VV in dB with detections and the reference (AIS or truth): matched, unexplained, explained, missed."""
    style()
    fig, ax = plt.subplots(figsize=figsize)
    with np.errstate(divide="ignore", invalid="ignore"):
        ax.imshow(10 * np.log10(scene.vv), cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    if water is not None:
        ax.contour(water, [0.5], colors=[AQUA], linewidths=0.4, alpha=0.6)
    d, v = val.dets, val.vessels
    expl = d[~d.matched & ~d.unexplained]
    une = d[d.unexplained]
    ax.scatter(d.col[d.matched], d.row[d.matched], s=70, facecolors="none", edgecolors=AQUA, lw=1.2,
               label=f"detection matched to {ref} ({int(d.matched.sum())})")
    ax.scatter(une.col, une.row, s=40, marker="s", facecolors="none", edgecolors=ORANGE, lw=1.0,
               label=f"unexplained detection ({len(une)})")
    ax.scatter(expl.col, expl.row, s=14, marker="x", color="#b9b8b2", lw=0.8,
               label=f"fragment, sidelobe, ambiguity or near a structure ({len(expl)})")
    miss = v[~v.detected]
    ax.scatter(miss.col, miss.row, s=60, marker="+", color=BLUE, lw=1.4,
               label=f"{ref} not detected ({len(miss)})")
    ax.set(title=title, xticks=[], yticks=[])
    ax.grid(False)
    ax.legend(loc="lower left", fontsize=8, facecolor=SURFACE, framealpha=0.9, frameon=True)
    fig.tight_layout()
    return fig


def plot_chips(scene, dets: pd.DataFrame, title: str, n: int = 8, half: int = 25):
    """VV and VH chips centred on detections (top row VV, bottom row VH)."""
    style()
    dets = dets.head(n)
    k = max(len(dets), 1)
    fig, axs = plt.subplots(2, k, figsize=(1.6 * k + 0.6, 3.6), squeeze=False)
    for j in range(k):
        for i in range(2):
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([]); axs[i, j].grid(False)
    for j, (_, t) in enumerate(dets.iterrows()):
        r, c = int(round(t.row)), int(round(t.col))
        for i, (img, lo) in enumerate([(scene.vv, -25.0), (scene.vh, -30.0)]):
            if img is None:
                continue
            crop = img[max(r - half, 0):r + half + 1, max(c - half, 0):c + half + 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                axs[i, j].imshow(10 * np.log10(crop), cmap="gray", vmin=lo, vmax=lo + 25)
        axs[0, j].set_title(f"{t.lat:.3f}N {abs(t.lon):.3f}W\n{t.length_psf_m:.0f} m, {t.peak_db_over_clutter:.0f} dB",
                            fontsize=7)
    axs[0, 0].set_ylabel("VV"); axs[1, 0].set_ylabel("VH")
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Real-data validation
# --------------------------------------------------------------------------- #
def plot_pd_length(tables: dict[str, pd.DataFrame]):
    """Detection probability by AIS length bin with Wilson 95 % intervals, per detector stage."""
    style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    first = next(iter(tables.values()))
    bins = [b for b in first.length_bin if b not in ("all", "unknown")]
    bins = [b for b in bins if first.set_index("length_bin").loc[b, "n_vessels"] > 0]
    x = np.arange(len(bins))
    w = 0.32
    for k, ((name, t), color, mk) in enumerate(zip(tables.items(), [BLUE, ORANGE, AQUA], "osD")):
        t = t.set_index("length_bin").loc[bins]
        xs = x + (k - (len(tables) - 1) / 2) * w
        err = np.clip([t.pd - t.ci_low, t.ci_high - t.pd], 0, None)
        ax.errorbar(xs, t.pd, yerr=err, fmt=mk, color=color, ms=6,
                    capsize=2, lw=1.2, label=name)
    n = first.set_index("length_bin").loc[bins, "n_vessels"].to_numpy()
    ax.set_xticks(x, [f"{b}\n(n={m})" for b, m in zip(bins, n)])
    ax.set(ylim=(-0.03, 1.05), ylabel="detection probability", xlabel="AIS length",
           title="Detection probability by vessel length vs AIS")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_length_vs_ais(est: np.ndarray, ais_len: np.ndarray, core: np.ndarray | None = None):
    style()
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    top = np.nanmax(np.r_[est, ais_len, core if core is not None else []]) * 1.1
    ax.plot([0, top], [0, top], color=INK2, lw=1, ls="--", label="1:1")
    if core is not None:
        ax.scatter(ais_len, core, s=14, color=ORANGE, marker="s", label="core extent")
    ax.scatter(ais_len, est, s=16, color=BLUE, marker="o", label="core, blur-corrected")
    ax.set(xlim=(0, top), ylim=(0, top), xlabel="AIS length (m)", ylabel="SAR length estimate (m)",
           title="Length: SAR vs AIS (matched vessels)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_clutter_tail(ratio: np.ndarray, L: float, nu: float, enl: float):
    """Empirical exceedance of I / local mean on sea pixels vs fitted clutter models."""
    style()
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    t = np.linspace(0.5, 25, 120)
    emp = np.array([(ratio > x).mean() for x in t])
    floor = 0.5 / ratio.size
    ax.plot(t, np.where(emp > 0, emp, np.nan), color=INK, lw=2, label=f"sea pixels ({ratio.size:,})")
    ax.plot(t, clutter.gamma_sf(t, L), color=BLUE, ls="--", label=f"gamma, L = {L:.1f}")
    ax.plot(t, clutter.gamma_sf(t, enl), color=AQUA, ls=":", label=f"gamma, measured ENL = {enl:.1f}")
    if np.isfinite(nu):
        ax.plot(t, [clutter.k_sf(x, L, nu) for x in t], color=ORANGE, ls="-.", label=f"K, L = {L:.1f}, nu = {nu:.1f}")
    ax.axhline(floor, color=GRID, lw=1)
    ax.set(yscale="log", ylim=(floor, 1.2), xlabel="intensity / local sea mean",
           ylabel="fraction of pixels above", title="Sea clutter tail vs model")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig
