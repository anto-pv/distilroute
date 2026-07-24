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
        # str(): classes_ is a numpy array, so classes_[idx] is a np.str_ that
        # would otherwise leak into caller state / graph nodes.
        return str(self.model.classes_[idx]), float(proba[idx])
