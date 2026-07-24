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
