"""Fusion stage: feature vector -> quality index.

RidgeFusion is the mandatory baseline (deterministic, closed-form).
GbtFusion wraps XGBoost with monotone constraints when available.
Per the plan: ship GBT only if it beats ridge by a meaningful margin on
content-grouped held-out data.
"""

import numpy as np


class RidgeFusion:
    """Standardized ridge regression, closed form."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self._mu = None
        self._sd = None
        self._w = None
        self._b = None

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._mu = x.mean(axis=0)
        self._sd = np.maximum(x.std(axis=0), 1e-9)
        xs = (x - self._mu) / self._sd
        d = xs.shape[1]
        a = xs.T @ xs + self.alpha * np.eye(d)
        self._b = float(y.mean())
        self._w = np.linalg.solve(a, xs.T @ (y - self._b))
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        xs = (x - self._mu) / self._sd
        return xs @ self._w + self._b

    def feature_weights(self, names, k=10):
        order = np.argsort(np.abs(self._w))[::-1][:k]
        return [(names[j], float(self._w[j])) for j in order]

    def to_dict(self):
        return {
            "alpha": self.alpha,
            "mu": self._mu.tolist(),
            "sd": self._sd.tolist(),
            "w": self._w.tolist(),
            "b": self._b,
        }

    @classmethod
    def from_dict(cls, d):
        m = cls(alpha=d["alpha"])
        m._mu = np.array(d["mu"], dtype=np.float64)
        m._sd = np.array(d["sd"], dtype=np.float64)
        m._w = np.array(d["w"], dtype=np.float64)
        m._b = float(d["b"])
        return m


class GbtFusion:
    """XGBoost wrapper with optional monotone constraints (optional dependency).

    monotone: dict {feature_name: +1|-1|0}; unknown-direction features get 0.
    """

    def __init__(self, names=None, monotone=None, **params):
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "GbtFusion requires xgboost (pip install xgboost). "
                "Use RidgeFusion as the dependency-free baseline."
            ) from exc
        import xgboost as xgb

        defaults = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
        )
        defaults.update(params)
        if names is not None and monotone:
            constraints = tuple(int(monotone.get(n, 0)) for n in names)
            defaults["monotone_constraints"] = constraints
        self.model = xgb.XGBRegressor(**defaults)

    def fit(self, x, y, eval_set=None):
        kwargs = {}
        if eval_set is not None:
            kwargs["eval_set"] = eval_set
            kwargs["verbose"] = False
        self.model.fit(np.asarray(x), np.asarray(y), **kwargs)
        return self

    def predict(self, x):
        return self.model.predict(np.asarray(x))
