from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.arp_redshift import half_time, solve_euler, z_of_t


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate ARP redshift law and generate plots")
    p.add_argument("--z0", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--tmax", type=float, default=8.0)
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--out", type=Path, default=Path("out"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0.0, args.tmax, args.n)
    z_exact = z_of_t(t, z0=args.z0, gamma=args.gamma)
    z_num = solve_euler(t, z_init=0.0, z0=args.z0, gamma=args.gamma)
    ode_residual = np.gradient(z_exact, t) - args.gamma * (args.z0 - z_exact)

    np.savetxt(
        out / "redshift_solution.csv",
        np.column_stack([t, z_exact, z_num, ode_residual]),
        delimiter=",",
        header="t,z_exact,z_euler,ode_residual",
        comments="",
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(t, z_exact, label="analytic z(t)")
    axes[0].plot(t, z_num, "--", label="euler z(t)", alpha=0.8)
    axes[0].axhline(args.z0, color="gray", linestyle=":", label="z0 asymptote")
    axes[0].axvline(half_time(args.gamma), color="purple", linestyle=":", label="t_1/2")
    axes[0].set_ylabel("z")
    axes[0].set_title("ARP redshift law: z(t) = z0 (1 - exp(-gamma t))")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, ode_residual, color="crimson", label="ODE residual")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("residual")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "redshift_solution.png", dpi=180)
    plt.close(fig)

    print(f"Saved {out / 'redshift_solution.csv'}")
    print(f"Saved {out / 'redshift_solution.png'}")
    print(f"Half-time t_1/2 = {half_time(args.gamma):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
