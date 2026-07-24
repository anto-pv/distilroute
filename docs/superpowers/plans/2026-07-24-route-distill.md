# route-distill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pip-installable plugin that distills an expensive LLM intent classifier into a cheap local model trained from the LLM's own logged decisions, diverting high-confidence traffic away from the LLM while never routing worse than LLM-only.

**Architecture:** Framework-agnostic core (log store → TF-IDF+LogisticRegression backend → trainer with a promotion gate → router with LLM fallback), wrapped by a `Distiller` facade. Optional duck-typed LangGraph node adapter and a cron-friendly CLI. The local model only handles a query when its confidence clears a threshold proven on a holdout to agree with the LLM ≥ target.

**Tech Stack:** Python 3.14, scikit-learn 1.8 (only hard dependency), stdlib `json`/`pickle`/`argparse`, pytest for tests, setuptools/pyproject for packaging.

## Global Constraints

- Python >= 3.10; only hard runtime dependency is `scikit-learn`.
- Core (`route_distill.core`, `route_distill.distiller`) MUST NOT import LangGraph. The adapter is duck-typed on a dict state.
- Log store format: one JSON object per line — keys `query` (str), `intent` (str), `confidence` (float).
- Safe fallback is inviolable: an untrained/unpromoted/failed model routes everything to the wrapped LLM.
- Package name: `route_distill`. CLI entry point: `distill`. License: MIT.
- Defaults from spec: `retrain_every=500`, `target_agreement=0.95`, `min_logs_to_train=200`, `backend=tfidf_lr`. Missing LLM confidence defaults to `1.0`.

---

## File Structure

```
route_distill/
  __init__.py          # exports Distiller
  distiller.py         # facade: wrap/route/train/report + threshold retrain trigger + persistence
  cli.py               # `distill train | report | status`
  core/
    __init__.py
    store.py           # append + read JSONL logs, skip corrupt lines
    backend.py         # TfidfLRBackend: fit / predict(text)->(intent, conf)
    trainer.py         # train_and_evaluate + promotion gate + format_report
  adapters/
    __init__.py
    langgraph.py       # make_router_node(distiller) -> callable node
tests/
  test_store.py
  test_backend.py
  test_trainer.py
  test_distiller.py
pyproject.toml
README.md
```

---

### Task 1: Log store (`core/store.py`)

**Files:**
- Create: `route_distill/__init__.py` (empty for now), `route_distill/core/__init__.py` (empty)
- Create: `route_distill/core/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `log_decision(path: str, query: str, intent: str, confidence: float) -> None`
  - `read_all(path: str) -> list[dict]` — each dict has `query`, `intent`, `confidence`; missing file returns `[]`; corrupt/blank lines skipped.
  - `count(path: str) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py
from route_distill.core import store

def test_roundtrip_and_skip_corrupt(tmp_path):
    p = str(tmp_path / "routes.jsonl")
    store.log_decision(p, "cancel my order", "cancel", 0.91)
    store.log_decision(p, "where is it", "track", 0.80)
    # inject a corrupt line
    with open(p, "a", encoding="utf-8") as f:
        f.write("{not valid json\n\n")
    rows = store.read_all(p)
    assert len(rows) == 2                     # corrupt + blank skipped
    assert rows[0] == {"query": "cancel my order", "intent": "cancel", "confidence": 0.91}
    assert store.count(p) == 2

def test_missing_file_is_empty(tmp_path):
    assert store.read_all(str(tmp_path / "none.jsonl")) == []
    assert store.count(str(tmp_path / "none.jsonl")) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'route_distill'`.

- [ ] **Step 3: Write minimal implementation**

```python
# route_distill/core/store.py
import json
from pathlib import Path


def log_decision(path, query, intent, confidence):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"query": query, "intent": intent,
                            "confidence": confidence}) + "\n")


def read_all(path):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # ponytail: skip corrupt line, never crash routing
    return rows


def count(path):
    return len(read_all(path))  # ponytail: O(n) reread; add tail-count if logs get huge
```

Also create empty `route_distill/__init__.py` and `route_distill/core/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_store.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add route_distill/__init__.py route_distill/core/__init__.py route_distill/core/store.py tests/test_store.py
git commit -m "feat: JSONL log store with corrupt-line skipping"
```

---

### Task 2: TF-IDF + LogisticRegression backend (`core/backend.py`)

**Files:**
- Create: `route_distill/core/backend.py`
- Test: `tests/test_backend.py`

**Interfaces:**
- Consumes: nothing (self-contained sklearn wrapper).
- Produces: class `TfidfLRBackend`
  - `fit(texts: list[str], labels: list[str]) -> None`
  - `predict(text: str) -> tuple[str, float]` — `(intent, confidence)`, confidence = max class probability.
  - attribute `.model` (the fitted sklearn Pipeline; used by `Distiller` for pickling).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backend.py
from route_distill.core.backend import TfidfLRBackend

def test_learns_separable_intents():
    texts = [
        "refund my money", "i want a refund", "give money back", "refund please",
        "track my package", "where is my shipment", "delivery status", "track order",
    ]
    labels = ["refund", "refund", "refund", "refund",
              "track", "track", "track", "track"]
    b = TfidfLRBackend()
    b.fit(texts, labels)
    intent, conf = b.predict("i need a refund now")
    assert intent == "refund"
    assert 0.0 <= conf <= 1.0
    assert conf > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'route_distill.core.backend'`.

- [ ] **Step 3: Write minimal implementation**

```python
# route_distill/core/backend.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


class TfidfLRBackend:
    """ponytail: TF-IDF + LogisticRegression. Pluggable later via same
    fit/predict interface if an embedding backend is ever needed."""

    def __init__(self):
        self.model = None

    def fit(self, texts, labels):
        self.model = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression(max_iter=1000),
        )
        self.model.fit(texts, labels)

    def predict(self, text):
        proba = self.model.predict_proba([text])[0]
        idx = proba.argmax()
        return self.model.classes_[idx], float(proba[idx])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_backend.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add route_distill/core/backend.py tests/test_backend.py
git commit -m "feat: TF-IDF + LogisticRegression backend"
```

---

### Task 3: Trainer + promotion gate + report (`core/trainer.py`)

**Files:**
- Create: `route_distill/core/trainer.py`
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `TfidfLRBackend` (Task 2).
- Produces:
  - `train_and_evaluate(logs: list[dict], target_agreement: float = 0.95, seed: int = 0) -> tuple[TfidfLRBackend, dict]`
    - returns `(backend, promotion)` where `promotion = {"threshold": float|None, "coverage": float, "agreement": float}`.
    - `threshold` is the lowest confidence at which holdout agreement with the LLM label ≥ `target_agreement` (maximizing coverage); `None` if no threshold qualifies.
  - `format_report(promotion: dict) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trainer.py
from route_distill.core import trainer

REFUND = ["refund my money", "i want a refund", "money back please", "refund now"]
TRACK  = ["track my package", "where is my order", "delivery status", "track shipment"]

def _logs(texts, intent):
    return [{"query": t, "intent": intent, "confidence": 1.0} for t in texts]

def test_promotes_when_separable():
    logs = (_logs(REFUND * 4, "refund") + _logs(TRACK * 4, "track"))
    _, promo = trainer.train_and_evaluate(logs, target_agreement=0.95, seed=0)
    assert promo["threshold"] is not None
    assert promo["coverage"] > 0.0
    assert promo["agreement"] >= 0.95
    assert "%" in trainer.format_report(promo)

def test_no_promotion_when_unlearnable():
    # same tiny vocab, labels alternate -> model cannot separate
    words = ["thing", "stuff", "item", "object"]
    logs = []
    for i in range(40):
        logs.append({"query": words[i % 4], "intent": ["a", "b"][i % 2],
                     "confidence": 1.0})
    _, promo = trainer.train_and_evaluate(logs, target_agreement=0.95, seed=0)
    assert promo["threshold"] is None
    assert "LLM" in trainer.format_report(promo)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trainer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'route_distill.core.trainer'`.

- [ ] **Step 3: Write minimal implementation**

```python
# route_distill/core/trainer.py
import random

from .backend import TfidfLRBackend


def _split(logs, seed, holdout_frac=0.2):
    logs = list(logs)
    random.Random(seed).shuffle(logs)
    n = max(1, int(len(logs) * holdout_frac))
    return logs[n:], logs[:n]  # train, holdout


def _best_threshold(preds, truths, target_agreement):
    # preds: list[(intent, conf)]; truths: list[intent] (the LLM's label)
    total = len(preds)
    best = {"threshold": None, "coverage": 0.0, "agreement": 0.0}
    if total == 0:
        return best
    # ascending thresholds: first one meeting target = lowest = max coverage
    for t in sorted({round(c, 2) for _, c in preds}):
        covered = [(p, y) for p, y in zip(preds, truths) if p[1] >= t]
        if not covered:
            continue
        agree = sum(1 for (pi, _), y in covered if pi == y) / len(covered)
        if agree >= target_agreement:
            return {"threshold": t, "coverage": len(covered) / total,
                    "agreement": agree}
    return best


def train_and_evaluate(logs, target_agreement=0.95, seed=0):
    train, holdout = _split(logs, seed)
    backend = TfidfLRBackend()
    backend.fit([r["query"] for r in train], [r["intent"] for r in train])
    preds = [backend.predict(r["query"]) for r in holdout]
    truths = [r["intent"] for r in holdout]
    promotion = _best_threshold(preds, truths, target_agreement)
    return backend, promotion


def format_report(promotion):
    t = promotion["threshold"]
    if t is None:
        return ("No confidence threshold meets the agreement target; "
                "all traffic goes to the LLM.")
    return (f"Local model can handle {promotion['coverage'] * 100:.0f}% of "
            f"traffic at confidence >= {t}, agreeing with the LLM "
            f"{promotion['agreement'] * 100:.0f}% of the time.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trainer.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add route_distill/core/trainer.py tests/test_trainer.py
git commit -m "feat: trainer with holdout promotion gate and report"
```

---

### Task 4: Distiller facade (`distiller.py`) + package export

**Files:**
- Create: `route_distill/distiller.py`
- Modify: `route_distill/__init__.py` (export `Distiller`)
- Test: `tests/test_distiller.py`

**Interfaces:**
- Consumes: `core.store` (Task 1), `core.trainer.train_and_evaluate` + return shape (Task 3), `TfidfLRBackend` (Task 2).
- Produces: class `Distiller`
  - `__init__(store="routes.jsonl", model_path="route_model.pkl", retrain_every=500, target_agreement=0.95, min_logs_to_train=200)`
  - `wrap(llm_fn) -> callable` — stores `llm_fn` (which takes `text` and returns `(intent, confidence)` or just `intent`) and returns `self.route`.
  - `route(text) -> tuple[str, str, float]` — `(intent, source, confidence)`, `source in {"local","llm"}`. Logs every LLM decision; fires threshold retrain.
  - `train() -> dict` — retrains from logs if `len(logs) >= min_logs_to_train`, updates `self.backend`/`self.threshold`/`self.promotion`, persists, returns promotion.
  - `report() -> str`
  - attributes: `.threshold` (float|None), `.promotion` (dict), `.backend`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_distiller.py
from route_distill import Distiller

REFUND = ["refund my money", "i want a refund", "money back please", "refund now"]
TRACK  = ["track my package", "where is my order", "delivery status", "track shipment"]

def _seed_store(path):
    from route_distill.core import store
    for t in REFUND * 4:
        store.log_decision(path, t, "refund", 1.0)
    for t in TRACK * 4:
        store.log_decision(path, t, "track", 1.0)

def test_untrained_falls_back_to_llm(tmp_path):
    d = Distiller(store=str(tmp_path / "r.jsonl"),
                  model_path=str(tmp_path / "m.pkl"))
    called = {"n": 0}
    def llm(text):
        called["n"] += 1
        return "track", 0.99
    route = d.wrap(llm)
    intent, source, conf = route("where is my stuff")
    assert (intent, source) == ("track", "llm")
    assert called["n"] == 1

def test_trained_routes_high_confidence_locally(tmp_path):
    sp = str(tmp_path / "r.jsonl")
    d = Distiller(store=sp, model_path=str(tmp_path / "m.pkl"),
                  min_logs_to_train=8, target_agreement=0.95)
    _seed_store(sp)
    promo = d.train()
    assert promo["threshold"] is not None
    def llm(text):
        raise AssertionError("LLM should not be called for a confident local hit")
    d.wrap(llm)
    intent, source, conf = d.route("i really need a refund")
    assert (intent, source) == ("refund", "local")

def test_threshold_trigger_calls_train(tmp_path):
    sp = str(tmp_path / "r.jsonl")
    d = Distiller(store=sp, model_path=str(tmp_path / "m.pkl"),
                  retrain_every=1, min_logs_to_train=1)
    hits = {"n": 0}
    orig = d.train
    def spy():
        hits["n"] += 1
        return orig()
    d.train = spy
    d.wrap(lambda t: ("track", 0.9))
    d.route("where is it")
    assert hits["n"] == 1  # retrain fired after threshold reached
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_distiller.py -v`
Expected: FAIL — `ImportError: cannot import name 'Distiller'`.

- [ ] **Step 3: Write minimal implementation**

```python
# route_distill/distiller.py
import pickle

from .core import store, trainer


class Distiller:
    def __init__(self, store="routes.jsonl", model_path="route_model.pkl",
                 retrain_every=500, target_agreement=0.95,
                 min_logs_to_train=200):
        self.store_path = store
        self.model_path = model_path
        self.retrain_every = retrain_every
        self.target_agreement = target_agreement
        self.min_logs_to_train = min_logs_to_train
        self.backend = None
        self.threshold = None
        self.promotion = {"threshold": None, "coverage": 0.0, "agreement": 0.0}
        self._llm_fn = None
        self._since_train = 0
        self._try_load()

    def _try_load(self):
        try:
            with open(self.model_path, "rb") as f:
                d = pickle.load(f)
            self.backend = d["backend"]
            self.promotion = d["promotion"]
            self.threshold = self.promotion["threshold"]
        except (FileNotFoundError, KeyError, pickle.UnpicklingError, EOFError):
            self.backend = None  # ponytail: any load failure => safe LLM-only

    def _save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump({"backend": self.backend, "promotion": self.promotion}, f)

    def wrap(self, llm_fn):
        self._llm_fn = llm_fn
        return self.route

    def route(self, text):
        if self.backend is not None and self.threshold is not None:
            intent, conf = self.backend.predict(text)
            if conf >= self.threshold:
                return intent, "local", conf
        res = self._llm_fn(text)
        intent, conf = res if isinstance(res, tuple) else (res, 1.0)
        store.log_decision(self.store_path, text, intent, conf)
        self._since_train += 1
        if (self.retrain_every and self._since_train >= self.retrain_every
                and store.count(self.store_path) >= self.min_logs_to_train):
            self.train()
        return intent, "llm", conf

    def train(self):
        logs = store.read_all(self.store_path)
        if len(logs) < self.min_logs_to_train:
            return self.promotion
        self.backend, self.promotion = trainer.train_and_evaluate(
            logs, target_agreement=self.target_agreement)
        self.threshold = self.promotion["threshold"]
        self._since_train = 0
        self._save()
        return self.promotion

    def report(self):
        return trainer.format_report(self.promotion)
```

```python
# route_distill/__init__.py
from .distiller import Distiller

__all__ = ["Distiller"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_distiller.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS (all tasks 1-4 green).

- [ ] **Step 6: Commit**

```bash
git add route_distill/distiller.py route_distill/__init__.py tests/test_distiller.py
git commit -m "feat: Distiller facade with LLM fallback, training, and threshold retrain"
```

---

### Task 5: CLI (`cli.py`) — train / report / status

**Files:**
- Create: `route_distill/cli.py`
- Test: none (thin argparse glue over already-tested `Distiller`; verified via manual run in Step 3).

**Interfaces:**
- Consumes: `Distiller` (Task 4), `core.store.count` (Task 1).
- Produces: `main(argv=None) -> int`. Subcommands `train`, `report`, `status`. Flags `--store` (default `routes.jsonl`), `--model` (default `route_model.pkl`). This is the cron-friendly entry point (scheduled retrain = cron running `distill train`).

- [ ] **Step 1: Write the implementation**

```python
# route_distill/cli.py
import argparse

from .core import store
from .distiller import Distiller


def main(argv=None):
    p = argparse.ArgumentParser(prog="distill")
    p.add_argument("command", choices=["train", "report", "status"])
    p.add_argument("--store", default="routes.jsonl")
    p.add_argument("--model", default="route_model.pkl")
    a = p.parse_args(argv)

    d = Distiller(store=a.store, model_path=a.model)
    if a.command == "train":
        d.train()
        print(d.report())
    elif a.command == "report":
        print(d.report())
    elif a.command == "status":
        print(f"{store.count(a.store)} logged decisions; "
              f"threshold={d.threshold}")
    return 0
```

- [ ] **Step 2: Verify it runs**

Run: `python -m route_distill.cli status --store nope.jsonl`
Expected: prints `0 logged decisions; threshold=None` and exits 0.

- [ ] **Step 3: Commit**

```bash
git add route_distill/cli.py
git commit -m "feat: cron-friendly CLI (train/report/status)"
```

---

### Task 6: LangGraph adapter (`adapters/langgraph.py`)

**Files:**
- Create: `route_distill/adapters/__init__.py` (empty)
- Create: `route_distill/adapters/langgraph.py`
- Test: `tests/test_distiller.py` (add one adapter test here to avoid a new file)

**Interfaces:**
- Consumes: any object with `.route(text) -> (intent, source, confidence)` (a `Distiller`).
- Produces: `make_router_node(distiller, input_key="input", output_key="intent") -> callable`
  - returned node: `node(state: dict) -> dict` reading `state[input_key]`, returning `{output_key: intent, "route_source": source, "route_confidence": conf}`.
- Note: no `langgraph` import — duck-typed on a dict state, so core stays dependency-free (Global Constraint).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_distiller.py
from route_distill.adapters.langgraph import make_router_node

class _FakeDistiller:
    def route(self, text):
        return "refund", "local", 0.97

def test_langgraph_node_maps_state():
    node = make_router_node(_FakeDistiller())
    out = node({"input": "refund please"})
    assert out == {"intent": "refund", "route_source": "local",
                   "route_confidence": 0.97}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_distiller.py::test_langgraph_node_maps_state -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'route_distill.adapters.langgraph'`.

- [ ] **Step 3: Write minimal implementation**

```python
# route_distill/adapters/langgraph.py
def make_router_node(distiller, input_key="input", output_key="intent"):
    """Return a LangGraph-compatible node. Duck-typed on dict state so this
    module never imports langgraph (keeps core dependency-free)."""
    def node(state):
        intent, source, conf = distiller.route(state[input_key])
        return {output_key: intent, "route_source": source,
                "route_confidence": conf}
    return node
```

Also create empty `route_distill/adapters/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_distiller.py::test_langgraph_node_maps_state -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add route_distill/adapters/__init__.py route_distill/adapters/langgraph.py tests/test_distiller.py
git commit -m "feat: duck-typed LangGraph router node adapter"
```

---

### Task 7: Packaging + README

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `LICENSE` (MIT)

**Interfaces:**
- Consumes: the whole `route_distill` package.
- Produces: an installable distribution exposing the `distill` console script and importable `route_distill.Distiller`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "route-distill"
version = "0.1.0"
description = "Distill an LLM intent classifier into a cheap local router trained from its own logs."
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.10"
dependencies = ["scikit-learn>=1.3"]

[project.optional-dependencies]
dev = ["pytest>=7"]

[project.scripts]
distill = "route_distill.cli:main"

[tool.setuptools.packages.find]
include = ["route_distill*"]
```

- [ ] **Step 2: Write `README.md`**

````markdown
# route-distill

Divert traffic away from an expensive LLM intent classifier. `route-distill`
logs the LLM's own decisions, trains a cheap local TF-IDF+LogisticRegression
model from them, and routes high-confidence traffic locally — while never
routing worse than LLM-only.

## Install

```bash
pip install route-distill
```

## Use

```python
from route_distill import Distiller

d = Distiller(store="routes.jsonl", retrain_every=500)
route = d.wrap(my_llm_classify_fn)   # my_llm_classify_fn(text) -> (intent, confidence)

intent, source, conf = route("cancel my order")   # source: "local" | "llm"

d.train()            # manual retrain (also fires automatically every retrain_every logs)
print(d.report())    # "Local model can handle 61% of traffic at confidence >= 0.9, ..."
```

### LangGraph

```python
from route_distill.adapters.langgraph import make_router_node
node = make_router_node(d)   # drop into your graph in place of the classifier node
```

### CLI (cron-friendly)

```bash
distill status     # how many decisions logged, current threshold
distill train      # retrain and print the coverage report
distill report     # print the last coverage report
```

## How it stays safe

The local model only answers when its confidence clears a threshold that was
proven on a holdout to agree with the LLM at least `target_agreement` (default
95%) of the time. Everything else falls back to the LLM. Worst case = today.
````

- [ ] **Step 3: Write `LICENSE`** — standard MIT license text, year 2026, author from git config.

- [ ] **Step 4: Verify install + entry point + full suite**

```bash
pip install -e .
distill status --store nope.jsonl
python -m pytest -v
```
Expected: editable install succeeds; `distill` prints `0 logged decisions; threshold=None`; all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml README.md LICENSE
git commit -m "chore: package for PyPI with distill CLI entry point and README"
```

---

## Self-Review

**Spec coverage:**
- Log collection by wrapping the LLM → Task 4 `route()` logs every LLM decision. ✅
- JSONL store, skip corrupt lines → Task 1. ✅
- TF-IDF+LR backend, pluggable → Task 2 (interface isolated for future embedding backend). ✅
- Trainer + holdout + promotion gate + report → Task 3. ✅
- Router with safe LLM fallback → Task 4 `route()`. ✅
- Retrain triggers: threshold (Task 4 auto), manual (`d.train()` Task 4), scheduled (CLI `distill train` under cron, Task 5). ✅
- LangGraph adapter, core dependency-free → Task 6. ✅
- Config defaults (`retrain_every`, `target_agreement`, `min_logs_to_train`) → Task 4 constructor. ✅
- Missing LLM confidence defaults to 1.0 → Task 4 `route()` (`res if isinstance(res, tuple) else (res, 1.0)`). ✅
- Error handling: corrupt log skipped (Task 1), load failure → LLM-only (Task 4 `_try_load`). ✅
- Packaging: PyPI, sklearn-only dep, MIT, README demo → Task 7. ✅
- Testing plan items (store round-trip incl. corrupt, backend separable, trainer promotion both ways, router fallback) → Tasks 1-4. ✅

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✅

**Type consistency:** `predict -> (intent, conf)` used identically in Tasks 2/3/4; `promotion` dict shape (`threshold`/`coverage`/`agreement`) consistent across Tasks 3/4; `route -> (intent, source, conf)` consistent across Tasks 4/6. ✅

**Deferred (YAGNI, per spec non-goals):** embedding backend, SQLite store backend, log rotation — interfaces left clean for later, not built.
