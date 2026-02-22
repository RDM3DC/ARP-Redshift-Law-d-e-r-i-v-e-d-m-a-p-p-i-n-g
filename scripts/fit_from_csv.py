from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.arp_redshift import estimate_params_from_timeseries, z_of_t


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit z0,gamma to z(t) observations")
    p.add_argument("--csv", type=Path, required=True, help="CSV with columns t,z (header optional)")
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--skip-header", action="store_true")
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
    params, mse = estimate_params_from_timeseries(t, z)
    z_fit = z_of_t(t, z0=params.z0, gamma=params.gamma)

    report = out / "fit_report.txt"
    report.write_text(
        "\n".join(
            [
                "ARP Redshift Fit Report",
                f"input={args.csv}",
                f"z0={params.z0:.10f}",
                f"gamma={params.gamma:.10f}",
                f"mse={mse:.12e}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    np.savetxt(
        out / "fit_series.csv",
        np.column_stack([t, z, z_fit, z - z_fit]),
        delimiter=",",
        header="t,z_obs,z_fit,residual",
        comments="",
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].scatter(t, z, s=20, label="observed")
    axes[0].plot(t, z_fit, color="darkorange", label="fit")
    axes[0].set_ylabel("z")
    axes[0].set_title("ARP redshift fit")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, z - z_fit, color="crimson", label="residual")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("obs - fit")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "fit_plot.png", dpi=180)
    plt.close(fig)

    print(report.read_text(encoding="utf-8"))
    print(f"Saved {out / 'fit_series.csv'}")
    print(f"Saved {out / 'fit_plot.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
