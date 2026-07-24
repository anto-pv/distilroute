# Contributing to distilroute

Thanks for your interest in improving distilroute! This is a small, focused
project — a drop-in router that distills an LLM intent classifier into a cheap
local model trained from the LLM's own logs. Contributions of all sizes are
welcome, from typo fixes to new backends.

## Ways to contribute

- **Report a bug** — open an issue with a minimal reproduction.
- **Suggest a feature** — open an issue describing the problem first, not just
  the solution. The most wanted item is an **embedding backend** (the
  `backend.py` interface was designed for exactly this).
- **Improve docs** — README, docstrings, examples.
- **Send a pull request** — see below.

## Development setup

```bash
git clone git@github.com:anto-pv/distilroute.git
cd distilroute
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest -v      # should be all green
```

Requires Python >= 3.10. The only hard dependency is scikit-learn.

## Project layout

```
distilroute/
  core/store.py      # JSONL decision log (append + read)
  core/backend.py    # TfidfLRBackend — fit / predict(text) -> (intent, conf)
  core/trainer.py    # training + holdout promotion gate + report
  distiller.py       # Distiller facade: wrap / route / train / report
  adapters/langgraph.py
  cli.py             # distill train | report | status
tests/               # one test file per module
```

Design notes and the original spec live in `docs/superpowers/`.

## How we work

This project was built test-first and reviewed task-by-task, and we'd like to
keep that bar. A few principles:

- **Tests first.** Every behavior change ships with a test that fails before
  your change and passes after. No frameworks beyond pytest, no elaborate
  fixtures.
- **Safe fallback is inviolable.** A data or training fault must NEVER crash
  live routing or package construction — the worst case is always "fall back
  to the LLM." If your change touches `route()`, `train()`, `_try_load`, or the
  log store, keep this invariant and add a test that proves it.
- **Keep the core dependency-free.** `distilroute.core` and `distilroute.distiller`
  must not import LangGraph (or any framework). Framework glue goes in
  `adapters/`.
- **Small and focused.** Prefer the smallest change that works over speculative
  abstraction. One clear responsibility per file.

## Pull request checklist

Before opening a PR, please make sure:

- [ ] `python -m pytest -v` passes locally.
- [ ] New behavior has a test (RED → GREEN).
- [ ] Public API changes are reflected in the README.
- [ ] The safe-fallback invariant still holds if you touched the routing path.
- [ ] Commits have clear messages (imperative mood, e.g. "add embedding backend").

Open the PR against `main` with a short description of the problem it solves.
Small, single-purpose PRs get reviewed fastest.

## Reporting security issues

If you find a security issue, please email antopv833@gmail.com rather than
opening a public issue.

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
