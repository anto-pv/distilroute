# route-distill — Design Spec

**Date:** 2026-07-24
**Status:** Approved for planning

## Problem

Chatbots built on LangGraph (and similar) call an expensive LLM intent
classifier on *every* turn. Most of that traffic is easy, repetitive, and
doesn't need an LLM. We want to divert that traffic to a cheap local model —
without hand-labeling training data, and without ever making routing *worse*
than today.

## Idea

A drop-in plugin that **distills** your existing LLM classifier into a cheap
local model, trained from the LLM's *own past decisions*. The plugin:

1. Wraps your LLM classifier and logs its decisions (free, as a side effect).
2. Trains a cheap local model from those logs.
3. Routes high-confidence, LLM-agreeing traffic locally; everything else still
   goes to the LLM.

Worst case = same behavior as today. Best case = the LLM sees a fraction of
the traffic.

## Goals

- Zero manual labeling — training data comes from the LLM's own logs.
- Never route worse than the current LLM-only setup (safe fallback).
- Drop-in: one wrap call for any classifier; a ready-made node for LangGraph.
- Measurable: report how much traffic can be safely handled locally.

## Non-Goals

- Not a general ML framework. One job: distill an intent classifier.
- Not replacing the LLM — it stays as the fallback and the source of truth.
- No embedding backend at v1 (structured so it can be added later — YAGNI).

## Architecture

Framework-agnostic core + optional thin LangGraph adapter.

```
route-distill/
  core/
    logger.py       # append (query, intent, confidence) to store
    store.py        # JSONL read/write (SQLite is a future backend)
    backend.py      # pluggable classifier interface
    tfidf_lr.py     # default backend: TF-IDF + LogisticRegression
    trainer.py      # train from logs, hold-out validation, promotion gate
    router.py       # route(text) -> (intent, source, confidence)
    distiller.py    # public API: wrap(), route(), train(), report()
  adapters/
    langgraph.py    # ready-made LangGraph node
  cli.py            # `distill train | report | status`
```

### Components (each independently testable)

- **logger / store** — appends decisions to a JSONL file. Interface:
  `log(query, intent, confidence)`, `read_all()`. SQLite is a drop-in future
  backend behind the same interface.
- **backend** — abstract: `fit(X, y)`, `predict(text) -> (intent, confidence)`,
  `save/load`. Default impl `tfidf_lr` (scikit-learn TF-IDF vectorizer +
  LogisticRegression). Pluggable so an embedding backend can be added without
  touching the rest.
- **trainer** — reads logs, splits train/holdout, fits the backend, measures
  agreement with LLM labels on the holdout, and computes the **promotion
  report**: at each confidence threshold, what % of traffic the local model
  would take and what % it agrees with the LLM there. Chooses the confidence
  threshold that meets the target agreement.
- **router** — at inference: run local backend; if confidence >= promoted
  threshold, return local intent (`source="local"`); else call the wrapped LLM
  (`source="llm"`) and log that decision.
- **distiller** — the public facade tying it together.

## Data Flow

**Learning phase (cold start):** local model untrained → every query goes to
the LLM → each decision logged. No behavior change, logs accumulate.

**After training:** query → local backend → high confidence? return local
intent : call LLM (and log it). LLM traffic keeps feeding new logs, so the
model keeps improving on later retrains.

## Public API

```python
from route_distill import Distiller

d = Distiller(store="routes.jsonl", retrain_every=500)

# wrap your existing LLM classifier once
route = d.wrap(my_llm_classify_fn)   # my_llm_classify_fn(text) -> (intent, confidence)

intent, source, conf = route("cancel my order")   # source: "local" | "llm"

d.train()        # manual retrain
print(d.report())  # agreement/coverage table
```

LangGraph:

```python
from route_distill.adapters.langgraph import make_router_node
node = make_router_node(d)   # drop into your graph in place of the classifier node
```

## Retrain Triggers (all three supported)

1. **Threshold (default)** — auto-retrain after every `retrain_every` new logs.
2. **Manual** — `d.train()` or `distill train` CLI.
3. **Scheduled** — user wires `distill train` into cron/Task Scheduler. We don't
   run a daemon; we just make the CLI cron-friendly.

## Safety / Promotion Gate

The local model is only allowed to handle a query when:

- its confidence >= the promoted threshold, AND
- on the holdout, at that threshold it agrees with the LLM >= `target_agreement`
  (default 0.95).

If no threshold meets the target, the local model handles nothing → pure LLM
fallback → identical to today. Promotion is recomputed on every retrain.

## Config

```yaml
store: routes.jsonl
retrain_every: 500          # threshold trigger; 0 disables auto-retrain
target_agreement: 0.95      # promotion gate
backend: tfidf_lr           # future: embedding
min_logs_to_train: 200      # don't train on too little data
```

## Error Handling

- Corrupt/partial log line → skip it, warn, keep going (don't crash routing).
- Backend load failure or untrained model → fall back to LLM (safe default).
- LLM call fails → propagate (that's the caller's existing failure mode).

## Testing

Small assert-based tests, no framework overhead:

- `store` round-trips logs incl. a corrupt line (skipped, not fatal).
- `tfidf_lr` learns a trivially separable 2-intent set and predicts correctly.
- `trainer` promotion gate: synthetic logs where local perfectly matches LLM →
  promotes; where it's random → promotes nothing.
- `router` fallback: untrained model routes everything to the (mock) LLM.

## Distribution

- **PyPI:** `pip install route-distill`.
- Framework-agnostic core; scikit-learn is the only hard dependency.
- Optional LangGraph adapter (import guarded so core works without LangGraph).
- MIT license, GitHub, README with a before/after "cut N% of classifier calls"
  demo.

## Milestones

1. Core: store + tfidf_lr backend + router with LLM fallback.
2. Trainer + promotion gate + `report()`.
3. Retrain triggers (threshold/manual/CLI).
4. LangGraph adapter.
5. Package for PyPI + README demo.

## Open Questions

- Confidence from the LLM classifier: some return no score. Fallback: treat
  missing LLM confidence as 1.0 (trust the LLM label fully for training).
- Log rotation/size at very high volume — defer until it's a real problem.
