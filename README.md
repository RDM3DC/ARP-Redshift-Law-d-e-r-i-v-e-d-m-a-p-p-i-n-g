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
- `scripts/compare_models.py`: compare single- vs two-timescale fits with AIC/BIC.
- `scripts/batch_compare_models.py`: rank many CSV tracks by single vs two-timescale evidence.
- `scripts/sim_oscillatory.py`: simulate and scan oscillatory extension parameters.
- `scripts/sim_bounded_oscillatory.py`: bounded oscillatory law (nonnegative by construction for `0<=epsilon<1`).
- `data/sample_observations.csv`: example fit input.
- `data/candidate_top_equation.json`: leaderboard-ready candidate metadata.

## Quick Start

From repository root:

```bash
python scripts/solve_and_plot.py --z0 1.0 --gamma 0.8 --tmax 8 --out out
python scripts/fit_from_csv.py --csv data/sample_observations.csv --out out --skip-header
python scripts/compare_models.py --csv data/sample_observations.csv --out out --skip-header
python scripts/batch_compare_models.py --data-dir data --pattern "*.csv" --out out --skip-header
python scripts/sim_oscillatory.py --gamma 1 --omega 2 --z-horizon 10 --tmax 5 --out out
python scripts/sim_oscillatory.py --gamma 1 --omega 2 --z-horizon 10 --tmax 5 --scan --out out
python scripts/sim_bounded_oscillatory.py --gamma 1 --omega 2 --epsilon 0.35 --phi 0.4 --z-horizon 10 --tmax 5 --out out
```

Outputs are written to `out/`:

- `redshift_solution.csv`
- `redshift_solution.png`
- `fit_report.txt`
- `fit_series.csv`
- `fit_plot.png`
- `model_compare_report.txt`
- `model_compare_series.csv`
- `model_compare_plot.png`
- `batch_model_compare_report.txt`
- `batch_model_compare_summary.csv`
- `batch_model_compare_scores.png`
- `oscillatory_report.txt`
- `oscillatory_series.csv`
- `oscillatory_plot.png`
- `oscillatory_scan_report.txt`
- `oscillatory_scan_top.csv`
- `bounded_oscillatory_report.txt`
- `bounded_oscillatory_series.csv`
- `bounded_oscillatory_plot.png`

## Next practical step

Export `(t, z)` points from your animation frames and run:

```bash
python scripts/fit_from_csv.py --csv path/to/your_points.csv --out out --skip-header
```

That returns best-fit `z0, gamma` and residual plots so you can decide whether
single-timescale ARP relaxation is sufficient, or whether to extend to
two-timescale relaxation.
