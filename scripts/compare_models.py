from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.arp_redshift import (  # noqa: E402
    estimate_params_from_timeseries,
    estimate_two_timescale_params_from_timeseries,
    information_criteria,
    z_of_t,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare single vs two-timescale ARP redshift models")
    p.add_argument("--csv", type=Path, required=True, help="CSV with columns t,z")
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--skip-header", action="store_true")
    p.add_argument("--gamma-max", type=float, default=5.0)
    p.add_argument("--gamma-grid", type=int, default=120)
    return p.parse_args()


def load_tz(path: Path, skip_header: bool) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter=",", skiprows=1 if skip_header else 0)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        raise ValueError("input must have at least 2 columns: t,z")
    t = arr[:, 0].astype(float)
    z = arr[:, 1].astype(float)
    idx = np.argsort(t)
    return t[idx], z[idx]


def main() -> int:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    t, z = load_tz(args.csv, skip_header=args.skip_header)

    p1, mse1 = estimate_params_from_timeseries(t, z)
    z1 = z_of_t(t, z0=p1.z0, gamma=p1.gamma)
    rss1 = float(np.sum((z - z1) ** 2))
    ic1 = information_criteria(rss=rss1, n=t.size, k=2)

    p2, mse2 = estimate_two_timescale_params_from_timeseries(
        t,
        z,
        gamma_max=args.gamma_max,
        gamma_grid_size=args.gamma_grid,
    )
    z2 = p2.w1 * (1.0 - np.exp(-p2.gamma1 * t)) + p2.w2 * (1.0 - np.exp(-p2.gamma2 * t))
    rss2 = float(np.sum((z - z2) ** 2))
    ic2 = information_criteria(rss=rss2, n=t.size, k=4)

    winner = "two-timescale" if ic2["bic"] < ic1["bic"] else "single-timescale"

    report = out / "model_compare_report.txt"
    report.write_text(
        "\n".join(
            [
                "ARP Redshift Model Comparison",
                f"input={args.csv}",
                "",
                "Single-timescale model:",
                f"  z0={p1.z0:.10f}",
                f"  gamma={p1.gamma:.10f}",
                f"  mse={mse1:.12e}",
                f"  rss={rss1:.12e}",
                f"  aic={ic1['aic']:.6f}",
                f"  bic={ic1['bic']:.6f}",
                "",
                "Two-timescale model:",
                f"  z0={p2.z0:.10f}",
                f"  gamma1={p2.gamma1:.10f}",
                f"  gamma2={p2.gamma2:.10f}",
                f"  a={p2.a:.10f}",
                f"  w1={p2.w1:.10f}",
                f"  w2={p2.w2:.10f}",
                f"  mse={mse2:.12e}",
                f"  rss={rss2:.12e}",
                f"  aic={ic2['aic']:.6f}",
                f"  bic={ic2['bic']:.6f}",
                "",
                f"winner_by_bic={winner}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    np.savetxt(
        out / "model_compare_series.csv",
        np.column_stack([t, z, z1, z2, z - z1, z - z2]),
        delimiter=",",
        header="t,z_obs,z_single,z_double,res_single,res_double",
        comments="",
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].scatter(t, z, s=20, label="observed")
    axes[0].plot(t, z1, label="single-timescale", color="darkorange")
    axes[0].plot(t, z2, label="two-timescale", color="teal")
    axes[0].set_ylabel("z")
    axes[0].set_title("ARP redshift model comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, z - z1, label="single residual", color="crimson")
    axes[1].plot(t, z - z2, label="double residual", color="purple")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("obs - fit")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "model_compare_plot.png", dpi=180)
    plt.close(fig)

    print(report.read_text(encoding="utf-8"))
    print(f"Saved {out / 'model_compare_series.csv'}")
    print(f"Saved {out / 'model_compare_plot.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
