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
    p = argparse.ArgumentParser(
        description="Batch-compare single vs two-timescale ARP redshift models over CSV files"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Folder with CSV tracks")
    p.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern for CSV files")
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--skip-header", action="store_true")
    p.add_argument("--gamma-max", type=float, default=5.0)
    p.add_argument("--gamma-grid", type=int, default=120)
    p.add_argument("--top", type=int, default=20, help="Max tracks in score plot")
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


def analyze_track(path: Path, skip_header: bool, gamma_max: float, gamma_grid: int) -> dict[str, float | str]:
    t, z = load_tz(path, skip_header=skip_header)

    p1, mse1 = estimate_params_from_timeseries(t, z)
    z1 = z_of_t(t, z0=p1.z0, gamma=p1.gamma)
    rss1 = float(np.sum((z - z1) ** 2))
    ic1 = information_criteria(rss=rss1, n=t.size, k=2)

    p2, mse2 = estimate_two_timescale_params_from_timeseries(
        t,
        z,
        gamma_max=gamma_max,
        gamma_grid_size=gamma_grid,
    )
    z2 = p2.w1 * (1.0 - np.exp(-p2.gamma1 * t)) + p2.w2 * (1.0 - np.exp(-p2.gamma2 * t))
    rss2 = float(np.sum((z - z2) ** 2))
    ic2 = information_criteria(rss=rss2, n=t.size, k=4)

    delta_bic = ic1["bic"] - ic2["bic"]
    delta_aic = ic1["aic"] - ic2["aic"]
    winner = "two-timescale" if delta_bic > 0 else "single-timescale"

    return {
        "file": path.name,
        "n": int(t.size),
        "single_mse": mse1,
        "single_bic": ic1["bic"],
        "single_aic": ic1["aic"],
        "single_z0": p1.z0,
        "single_gamma": p1.gamma,
        "double_mse": mse2,
        "double_bic": ic2["bic"],
        "double_aic": ic2["aic"],
        "double_z0": p2.z0,
        "double_gamma1": p2.gamma1,
        "double_gamma2": p2.gamma2,
        "double_a": p2.a,
        "delta_bic": delta_bic,
        "delta_aic": delta_aic,
        "winner": winner,
    }


def write_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(r[h]) for h in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(args.data_dir.glob(args.pattern))
    files = [f for f in files if f.is_file()]
    if not files:
        raise SystemExit(f"No files found under {args.data_dir} matching {args.pattern}")

    rows: list[dict[str, float | str]] = []
    for f in files:
        try:
            rows.append(
                analyze_track(
                    f,
                    skip_header=args.skip_header,
                    gamma_max=args.gamma_max,
                    gamma_grid=args.gamma_grid,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "file": f.name,
                    "n": 0,
                    "single_mse": np.nan,
                    "single_bic": np.nan,
                    "single_aic": np.nan,
                    "single_z0": np.nan,
                    "single_gamma": np.nan,
                    "double_mse": np.nan,
                    "double_bic": np.nan,
                    "double_aic": np.nan,
                    "double_z0": np.nan,
                    "double_gamma1": np.nan,
                    "double_gamma2": np.nan,
                    "double_a": np.nan,
                    "delta_bic": np.nan,
                    "delta_aic": np.nan,
                    "winner": f"error:{exc}",
                }
            )

    valid = [r for r in rows if isinstance(r["delta_bic"], (float, int)) and not np.isnan(r["delta_bic"])]
    valid_sorted = sorted(valid, key=lambda r: float(r["delta_bic"]), reverse=True)

    summary_csv = out / "batch_model_compare_summary.csv"
    write_summary_csv(summary_csv, valid_sorted)

    two_count = sum(1 for r in valid_sorted if r["winner"] == "two-timescale")
    single_count = sum(1 for r in valid_sorted if r["winner"] == "single-timescale")

    report = out / "batch_model_compare_report.txt"
    report_lines = [
        "ARP Batch Model Comparison",
        f"data_dir={args.data_dir}",
        f"pattern={args.pattern}",
        f"tracks_total={len(files)}",
        f"tracks_valid={len(valid_sorted)}",
        f"winner_two_timescale={two_count}",
        f"winner_single_timescale={single_count}",
        "",
        "Top tracks by delta_bic (single - double):",
    ]

    for row in valid_sorted[:10]:
        report_lines.append(
            f"  {row['file']}: delta_bic={float(row['delta_bic']):.6f}, winner={row['winner']}"
        )

    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    top_rows = valid_sorted[: max(1, min(args.top, len(valid_sorted)))]
    if top_rows:
        labels = [str(r["file"]) for r in top_rows]
        values = [float(r["delta_bic"]) for r in top_rows]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(values)), values, color=["teal" if v > 0 else "darkorange" for v in values])
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("delta_bic = bic_single - bic_double")
        ax.set_title("Batch ARP model preference by track")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "batch_model_compare_scores.png", dpi=180)
        plt.close(fig)

    print(report.read_text(encoding="utf-8"))
    print(f"Saved {summary_csv}")
    if top_rows:
        print(f"Saved {out / 'batch_model_compare_scores.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
