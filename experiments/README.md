# Experiments

One folder per experiment run: `<DECISION-ID>__<short-name>__<YYYY-MM-DD>/`.

Each folder contains:

- `config.yaml` — exact parameters, model ids and revisions, hardware notes
- `command.txt` — the exact command(s) run, so the run is reproducible
- `results.csv` or `results.json` — raw measurements, not summaries
- `notes.md` — what was being tested, what surprised you, what to do next

Rules:

- Change **one** variable per run. Same benchmark, same hardware.
- Record raw numbers. A conclusion without the numbers behind it is not reproducible.
- Bulk artifacts (weights, indexes) go in `artifacts/` and are gitignored. The measurements are committed.
- When a run settles a question, write the decision into `DECISIONS.md` and link the experiment folder.
