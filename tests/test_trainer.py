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
    # same query with contradictory labels -> model cannot separate
    logs = []
    for i in range(40):
        logs.append({"query": "thing", "intent": ["a", "b"][i % 2],
                     "confidence": 1.0})
    _, promo = trainer.train_and_evaluate(logs, target_agreement=0.95, seed=0)
    assert promo["threshold"] is None
    assert "LLM" in trainer.format_report(promo)
