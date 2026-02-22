from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.arp_redshift import z_of_t_bounded_oscillatory  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate bounded oscillatory redshift law")
    p.add_argument("--tmax", type=float, default=5.0)
    p.add_argument("--n", type=int, default=501)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--omega", type=float, default=2.0)
    p.add_argument("--epsilon", type=float, default=0.35)
    p.add_argument("--phi", type=float, default=0.4)
    p.add_argument("--z-horizon", type=float, default=10.0)
    p.add_argument("--out", type=Path, default=Path("out"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0.0, args.tmax, args.n)
    z = z_of_t_bounded_oscillatory(
        t,
        z_horizon=args.z_horizon,
        gamma=args.gamma,
        omega=args.omega,
        epsilon=args.epsilon,
        phi=args.phi,
    )

    np.savetxt(
        out / "bounded_oscillatory_series.csv",
        np.column_stack([t, z]),
        delimiter=",",
        header="t,z",
        comments="",
    )

    z_min = float(np.min(z))
    z_max = float(np.max(z))
    z_final = float(z[-1])

    report = out / "bounded_oscillatory_report.txt"
    report.write_text(
        "\n".join(
            [
                "Bounded Oscillatory Redshift Law",
                "z(t)=z_h(1-exp(-gamma t))*(1-epsilon*cos(omega t + phi))",
                f"gamma={args.gamma}",
                f"omega={args.omega}",
                f"epsilon={args.epsilon}",
                f"phi={args.phi}",
                f"z_horizon={args.z_horizon}",
                f"t_range=[0,{args.tmax}]",
                f"z_min={z_min:.6f}",
                f"z_max={z_max:.6f}",
                f"z_final={z_final:.6f}",
                f"nonnegative_over_interval={int(np.all(z >= -1e-12))}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, z, label="bounded oscillatory z(t)", color="teal")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Bounded oscillatory redshift law")
    ax.set_xlabel("t")
    ax.set_ylabel("z")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "bounded_oscillatory_plot.png", dpi=180)
    plt.close(fig)

    print(report.read_text(encoding="utf-8"))
    print(f"Saved {out / 'bounded_oscillatory_series.csv'}")
    print(f"Saved {out / 'bounded_oscillatory_plot.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
