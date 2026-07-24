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

from route_distill.adapters.langgraph import make_router_node

class _FakeDistiller:
    def route(self, text):
        return "refund", "local", 0.97

def test_langgraph_node_maps_state():
    node = make_router_node(_FakeDistiller())
    out = node({"input": "refund please"})
    assert out == {"intent": "refund", "route_source": "local",
                   "route_confidence": 0.97}
