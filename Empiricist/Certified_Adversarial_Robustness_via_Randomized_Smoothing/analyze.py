#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from randomized_smoothing.core import max_certifiable_radius

PALETTE = {0.12: "#7b3294", 0.25: "#2166ac", 0.50: "#b2182b", 1.00: "#1b7837"}
REFERENCE = Path("reference/cohen2019")


def load_reference() -> dict[float, pd.DataFrame]:
    out = {}
    for tsv in sorted(REFERENCE.glob("cohen_sigma_*.tsv")):
        sigma = float(tsv.stem.replace("cohen_sigma_", ""))
        df = pd.read_csv(tsv, sep="\t")
        df["abstain"] = (df["predict"] == -1).astype(int)
        secs = pd.to_numeric(df["time"], errors="coerce")
        if secs.isna().any():
            secs = pd.to_timedelta(df["time"], errors="coerce").dt.total_seconds()
        df["seconds"] = secs
        out[sigma] = df.drop(columns=["time"])
    return out


def certified_accuracy(df: pd.DataFrame, radii: np.ndarray) -> np.ndarray:
    ok = (df["correct"] == 1).to_numpy()
    rad = df["radius"].to_numpy()
    return np.array([np.mean(ok & (rad >= r)) for r in radii])


def load(results_dir: Path) -> dict[float, pd.DataFrame]:
    out = {}
    for csv in sorted(results_dir.glob("certify_sigma_*.csv")):
        sigma = float(csv.stem.replace("certify_sigma_", ""))
        out[sigma] = pd.read_csv(csv)
    if not out:
        raise SystemExit(f"no certify_sigma_*.csv found in {results_dir}")
    return out


def fig_reference_comparison(runs, reference, out: Path) -> None:
    if not reference:
        print(f"no reference data in {REFERENCE}; skipping the overlay")
        return
    radii = np.linspace(0, 3.0, 300)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for sigma in sorted(set(runs) & set(reference)):
        c = PALETTE.get(sigma, "#444")
        ax.plot(radii, certified_accuracy(runs[sigma], radii), lw=2, color=c,
                label=f"ours, $\\sigma$={sigma:.2f}")
        ax.plot(radii, certified_accuracy(reference[sigma], radii), lw=1.6,
                ls="--", color=c, alpha=.75,
                label=f"Cohen et al., $\\sigma$={sigma:.2f}")
    ax.set_xlabel(r"$\ell_2$ radius $r$"); ax.set_ylabel("certified accuracy")
    ax.set_title("This reproduction (solid) vs the authors' released results (dashed)")
    ax.set_xlim(0, 3.0); ax.set_ylim(0, 1.0); ax.grid(alpha=.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def fig_certified_accuracy(runs, args, out: Path) -> None:
    radii = np.linspace(0, 3.0, 300)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for sigma, df in sorted(runs.items()):
        c = PALETTE.get(sigma, "#444")
        ax.plot(radii, certified_accuracy(df, radii), lw=2, color=c,
                label=f"$\\sigma$ = {sigma:.2f}")
        ceil = max_certifiable_radius(sigma, args["n"], args["alpha"])
        ax.axvline(ceil, color=c, ls=":", lw=1, alpha=.7)
    ax.set_xlabel(r"$\ell_2$ radius $r$")
    ax.set_ylabel("certified accuracy")
    ax.set_title(f"CIFAR-10, ResNet-110, n={args['n']:,}, "
                 f"$\\alpha$={args['alpha']}\n(dotted: radius ceiling from n)")
    ax.set_xlim(0, 3.0); ax.set_ylim(0, 1.0)
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def fig_upper_envelope(runs, out: Path) -> None:
    radii = np.linspace(0, 3.0, 300)
    curves = {s: certified_accuracy(df, radii) for s, df in runs.items()}
    best = np.max(np.stack(list(curves.values())), axis=0)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for sigma, y in sorted(curves.items()):
        ax.plot(radii, y, lw=1, alpha=.45, color=PALETTE.get(sigma, "#444"),
                label=f"$\\sigma$ = {sigma:.2f}")
    ax.plot(radii, best, lw=2.5, color="k", label="upper envelope")
    ax.set_xlabel(r"$\ell_2$ radius $r$"); ax.set_ylabel("certified accuracy")
    ax.set_title("No single $\\sigma$ dominates: the curves cross")
    ax.set_xlim(0, 3.0); ax.set_ylim(0, 1.0); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def fig_abstain_radius(runs, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sigmas = sorted(runs)
    axes[0].bar([f"{s:.2f}" for s in sigmas],
                [runs[s]["abstain"].mean() for s in sigmas],
                color=[PALETTE.get(s, "#444") for s in sigmas])
    axes[0].set_xlabel(r"$\sigma$"); axes[0].set_ylabel("abstention rate")
    axes[0].set_title("Abstentions rise with noise")
    for s in sigmas:
        d = runs[s]
        r = d.loc[(d["correct"] == 1), "radius"]
        axes[1].hist(r, bins=40, histtype="step", lw=2,
                     color=PALETTE.get(s, "#444"), label=f"$\\sigma$={s:.2f}")
    axes[1].set_xlabel("certified radius (correct predictions)")
    axes[1].set_ylabel("images"); axes[1].legend(fontsize=8)
    axes[1].set_title("Radius distribution")
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def make_table(runs, args) -> pd.DataFrame:
    report_radii = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    rows = []
    for sigma, df in sorted(runs.items()):
        acc = certified_accuracy(df, np.array(report_radii))
        rows.append({"sigma": sigma,
                     **{f"r={r}": a for r, a in zip(report_radii, acc)},
                     "abstain": df["abstain"].mean(),
                     "mean_sec": df["seconds"].mean(),
                     "ceiling": max_certifiable_radius(sigma, args["n"], args["alpha"])})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("outputs"))
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--alpha", type=float, default=0.001)
    a = p.parse_args()

    runs = load(a.results_dir)
    reference = load_reference()
    args = {"n": a.n, "alpha": a.alpha}
    figs = a.results_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    fig_certified_accuracy(runs, args, figs / "02_certified_accuracy.png")
    fig_upper_envelope(runs, figs / "04_upper_envelope.png")
    fig_abstain_radius(runs, figs / "03_abstain_and_radius.png")
    fig_reference_comparison(runs, reference, figs / "01_reproduction_vs_paper.png")

    table = make_table(runs, args)
    table.to_csv(a.results_dir / "summary.csv", index=False)
    if reference:
        ref_table = make_table(reference, args)
        print("\nCohen et al., recomputed from their released per-image output:")
        print(ref_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("\nThis reproduction:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nfigures -> {figs}")


if __name__ == "__main__":
    main()
