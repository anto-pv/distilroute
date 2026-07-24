import json
from pathlib import Path


def log_decision(path, query, intent, confidence):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"query": query, "intent": intent,
                            "confidence": confidence}) + "\n")


_EXPECTED_KEYS = ("query", "intent", "confidence")


def read_all(path):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    # errors="ignore": a stray non-UTF-8 byte must never crash read_all/count
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # ponytail: skip corrupt line, never crash routing
        if not isinstance(row, dict) or not all(k in row for k in _EXPECTED_KEYS):
            continue  # valid JSON but not a row we recognize; skip, don't crash
        rows.append(row)
    return rows


def count(path):
    return len(read_all(path))  # ponytail: O(n) reread; add tail-count if logs get huge
