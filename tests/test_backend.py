from distilroute.core.backend import TfidfLRBackend

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
    assert type(intent) is str          # not numpy.str_ leaking into caller state
    assert 0.0 <= conf <= 1.0
    assert conf > 0.5
