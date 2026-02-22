# ARP Redshift Law Derived Mapping

Core model:

- `z(t) = z0 * (1 - exp(-gamma * t))`
- ODE form: `dz/dt = gamma * (z0 - z)`

This repo includes a compact toolkit to:

- solve and visualize the equation,
- numerically validate the ODE bridge,
- estimate `(z0, gamma)` from observed timeseries data.

## Files

- `ARPRedshiftLawBridge.mp4`: animation artifact.
- `src/arp_redshift.py`: analytic + numeric core functions.
- `scripts/solve_and_plot.py`: generate solution curves and residual checks.
- `scripts/fit_from_csv.py`: fit parameters from CSV observations.
- `data/sample_observations.csv`: example fit input.

## Quick Start

From repository root:

```bash
python scripts/solve_and_plot.py --z0 1.0 --gamma 0.8 --tmax 8 --out out
python scripts/fit_from_csv.py --csv data/sample_observations.csv --out out --skip-header
```

Outputs are written to `out/`:

- `redshift_solution.csv`
- `redshift_solution.png`
- `fit_report.txt`
- `fit_series.csv`
- `fit_plot.png`

## Next practical step

Export `(t, z)` points from your animation frames and run:

```bash
python scripts/fit_from_csv.py --csv path/to/your_points.csv --out out --skip-header
```

That returns best-fit `z0, gamma` and residual plots so you can decide whether
single-timescale ARP relaxation is sufficient, or whether to extend to
two-timescale relaxation.
