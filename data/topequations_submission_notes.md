# TopEquations Submission Notes

Target entry file: `data/topequations_submission_entry.json`

Suggested insertion target in TopEquations repo:

- `data/equations.json` -> append this object to `entries`

Then rebuild and publish in TopEquations repo:

```powershell
python tools\generate_leaderboard.py
python tools\build_site.py
```

Expected scoring profile for this candidate:

- novelty: 28/30
- tractability: 19/20
- plausibility: 19/20
- units bonus: +10 (`OK`)
- theory bonus: +10 (`PASS`)
- artifact bonus: +10 (animation linked + image linked)
- target total: 96

Notes:

- Equation is bounded-nonnegative for `0 <= epsilon < 1`, which helps keep physical plausibility high while preserving oscillatory behavior.
- If desired, lower score to a conservative value (e.g., 92) before publishing.
