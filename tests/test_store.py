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
