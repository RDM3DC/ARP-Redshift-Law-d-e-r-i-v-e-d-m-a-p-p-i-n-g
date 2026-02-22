from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.arp_redshift import z_of_t_oscillatory  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate oscillatory ARP redshift toy model")
    p.add_argument("--tmax", type=float, default=5.0)
    p.add_argument("--n", type=int, default=501)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--omega", type=float, default=2.0)
    p.add_argument("--phi", type=float, default=0.0)
    p.add_argument("--z-horizon", type=float, default=10.0)
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--scan", action="store_true", help="Grid scan for no-flip parameter candidates")
    p.add_argument("--gamma-min", type=float, default=0.3)
    p.add_argument("--gamma-max", type=float, default=2.0)
    p.add_argument("--gamma-steps", type=int, default=25)
    p.add_argument("--omega-min", type=float, default=0.05)
    p.add_argument("--omega-max", type=float, default=2.5)
    p.add_argument("--omega-steps", type=int, default=120)
    p.add_argument("--phi-min", type=float, default=0.0)
    p.add_argument("--phi-max", type=float, default=3.0)
    p.add_argument("--phi-steps", type=int, default=100)
    p.add_argument("--top-k", type=int, default=12)
    return p.parse_args()


def count_sign_flips(z: np.ndarray) -> int:
    s = np.sign(z)
    # ignore exact zeros by forward fill
    for i in range(1, s.size):
        if s[i] == 0:
            s[i] = s[i - 1]
    flips = np.sum(s[1:] * s[:-1] < 0)
    return int(flips)


def summarize_track(t: np.ndarray, z: np.ndarray) -> dict[str, float | int]:
    return {
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "z_final": float(z[-1]),
        "sign_flips": count_sign_flips(z),
        "crosses_zero": int(np.any(z < 0) and np.any(z > 0)),
    }


def run_scan(args: argparse.Namespace, t: np.ndarray, out: Path) -> None:
    gammas = np.linspace(args.gamma_min, args.gamma_max, args.gamma_steps)
    omegas = np.linspace(args.omega_min, args.omega_max, args.omega_steps)
    phis = np.linspace(args.phi_min, args.phi_max, args.phi_steps)

    rows: list[tuple[float, float, float, float, float, int]] = []
    for g in gammas:
        for w in omegas:
            for p in phis:
                z = z_of_t_oscillatory(t, z_horizon=args.z_horizon, gamma=g, omega=w, phi=p)
                flips = count_sign_flips(z)
                zmin = float(np.min(z))
                zfinal = float(z[-1])
                score = zfinal - 20.0 * max(0.0, -zmin) - 1.0 * flips
                rows.append((score, g, w, p, zfinal, flips))

    rows.sort(key=lambda r: r[0], reverse=True)

    # strict no-flip/no-negative candidates
    strict = [r for r in rows if r[5] == 0]

    lines = ["score,gamma,omega,phi,z_final,sign_flips"]
    for r in rows[: args.top_k]:
        lines.append(
            f"{r[0]:.8f},{r[1]:.8f},{r[2]:.8f},{r[3]:.8f},{r[4]:.8f},{r[5]}"
        )
    (out / "oscillatory_scan_top.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = [
        "Oscillatory Scan Report",
        f"grid={args.gamma_steps}x{args.omega_steps}x{args.phi_steps}",
        f"samples={len(rows)}",
        "",
    ]
    if strict:
        best = strict[0]
        report.extend(
            [
                "Best strict (no sign flips):",
                f"  gamma={best[1]:.6f}",
                f"  omega={best[2]:.6f}",
                f"  phi={best[3]:.6f}",
                f"  z_final={best[4]:.6f}",
                f"  score={best[0]:.6f}",
            ]
        )
    else:
        report.append("No strict no-flip candidate found in scan range.")

    (out / "oscillatory_scan_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0.0, args.tmax, args.n)
    z = z_of_t_oscillatory(
        t,
        z_horizon=args.z_horizon,
        gamma=args.gamma,
        omega=args.omega,
        phi=args.phi,
    )

    np.savetxt(
        out / "oscillatory_series.csv",
        np.column_stack([t, z]),
        delimiter=",",
        header="t,z",
        comments="",
    )

    stats = summarize_track(t, z)
    report_lines = [
        "Oscillatory ARP Simulation",
        f"gamma={args.gamma}",
        f"omega={args.omega}",
        f"phi={args.phi}",
        f"z_horizon={args.z_horizon}",
        f"t_range=[0,{args.tmax}]",
        f"z_min={stats['z_min']:.6f}",
        f"z_max={stats['z_max']:.6f}",
        f"z_final={stats['z_final']:.6f}",
        f"sign_flips={stats['sign_flips']}",
    ]
    (out / "oscillatory_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, z, label="z(t)")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Oscillatory redshift toy model")
    ax.set_xlabel("t")
    ax.set_ylabel("z")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "oscillatory_plot.png", dpi=180)
    plt.close(fig)

    print((out / "oscillatory_report.txt").read_text(encoding="utf-8"))
    print(f"Saved {out / 'oscillatory_series.csv'}")
    print(f"Saved {out / 'oscillatory_plot.png'}")

    if args.scan:
        run_scan(args, t, out)
        print((out / "oscillatory_scan_report.txt").read_text(encoding="utf-8"))
        print(f"Saved {out / 'oscillatory_scan_top.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
