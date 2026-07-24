# distilroute

Divert traffic away from an expensive LLM intent classifier. `distilroute`
logs the LLM's own decisions, trains a cheap local TF-IDF+LogisticRegression
model from them, and routes high-confidence traffic locally — while never
routing worse than LLM-only.

## Install

```bash
pip install distilroute
```

## Use

```python
from distilroute import Distiller

d = Distiller(store="routes.jsonl", retrain_every=500)
route = d.wrap(my_llm_classify_fn)   # my_llm_classify_fn(text) -> (intent, confidence)

intent, source, conf = route("cancel my order")   # source: "local" | "llm"

d.train()            # manual retrain (also fires automatically every retrain_every logs)
print(d.report())    # "Local model can handle 61% of traffic at confidence >= 0.9, ..."
```

### LangGraph

```python
from distilroute.adapters.langgraph import make_router_node
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
