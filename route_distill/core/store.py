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
