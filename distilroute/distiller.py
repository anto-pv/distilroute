# distilroute/distiller.py
import pickle

from .core import store, trainer


class Distiller:
    def __init__(self, store="routes.jsonl", model_path="route_model.pkl",
                 retrain_every=500, target_agreement=0.95,
                 min_logs_to_train=200):
        self.store_path = store
        self.model_path = model_path
        self.retrain_every = retrain_every
        self.target_agreement = target_agreement
        self.min_logs_to_train = min_logs_to_train
        self.backend = None
        self.threshold = None
        self.promotion = {"threshold": None, "coverage": 0.0, "agreement": 0.0}
        self._llm_fn = None
        self._since_train = 0
        self._try_load()

    def _try_load(self):
        try:
            with open(self.model_path, "rb") as f:
                d = pickle.load(f)
            self.backend = d["backend"]
            self.promotion = d["promotion"]
            self.threshold = self.promotion["threshold"]
        except Exception:
            # ponytail: any load failure (missing file, torn/incompatible
            # pickle, version-mismatched class refs, etc.) => safe LLM-only
            self.backend = None

    def _save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump({"backend": self.backend, "promotion": self.promotion}, f)

    def wrap(self, llm_fn):
        self._llm_fn = llm_fn
        return self.route

    def route(self, text):
        if self.backend is not None and self.threshold is not None:
            intent, conf = self.backend.predict(text)
            if conf >= self.threshold:
                return intent, "local", conf
        res = self._llm_fn(text)
        intent, conf = res if isinstance(res, tuple) else (res, 1.0)
        store.log_decision(self.store_path, text, intent, conf)
        self._since_train += 1
        try:
            # Retrain is best-effort: any failure here (bad data, a
            # stranded single-class split, disk I/O, ...) must never
            # affect the response we're already about to return.
            if (self.retrain_every and self._since_train >= self.retrain_every
                    and store.count(self.store_path) >= self.min_logs_to_train):
                self.train()
        except Exception:
            pass
        return intent, "llm", conf

    def train(self):
        try:
            logs = store.read_all(self.store_path)
            if len(logs) < self.min_logs_to_train:
                return self.promotion
            if len({r["intent"] for r in logs}) < 2:
                return self.promotion  # ponytail: can't fit a classifier on one class
            try:
                backend, promotion = trainer.train_and_evaluate(
                    logs, target_agreement=self.target_agreement)
            except Exception:
                # e.g. a rare class stranded alone by the holdout split leaves
                # a single-class training set and LogisticRegression.fit
                # raises. Keep whatever model was previously promoted.
                return self.promotion
            self.backend, self.promotion = backend, promotion
            self.threshold = self.promotion["threshold"]
            self._save()
            return self.promotion
        finally:
            # Reset on every exit path so a persistently-failing retrain
            # doesn't force a full store.count() reread on every route().
            self._since_train = 0

    def report(self):
        return trainer.format_report(self.promotion)
