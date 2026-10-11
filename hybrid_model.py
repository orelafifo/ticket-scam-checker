"""
hybrid_model.py - the decision model (Layer 2).

v2.4 (after the independent held-out test exposed overfitting):
  1. Uses ONLY the red-flag / good-sign features, not individual words.
  2. Learns their weights mainly from REAL examples (120 published-case rows,
     weighted x50), with the synthetic listings as background.
  3. Sign constraint: a red flag can only RAISE the risk, a good sign only LOWER it.
  4. L2 regularisation pulls every weight towards 0.

v2.5 (expert evidence tiers - "informative priors"):
  The only change to the learning rule: L2 regularisation now pulls each weight
  towards its EXPERT starting weight (hard stop 3.0, strong 2.5, moderate 1.5,
  weak 0.5, good sign -2.0 - see features.TIERS and TIER_SOURCES.md) instead of
  towards 0. With little real data the weights stay close to the expert tiers;
  where real data clearly disagrees, it moves them.
      v2.4 penalty: sum(w^2)           / (2 * C * n)
      v2.5 penalty: sum((w - prior)^2) / (2 * C * n)
  The strength C was chosen by source-grouped cross-validation (select_v25.py).
  v2.5 also adds one feature: very_cheap (price under 50% of face value).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from features import FLAG_COLUMNS, PROTECTIVE, V24_FLAG_COLUMNS

REAL_WEIGHT = 50     # chosen by grouped cross-validation in v2.4 (5 / 20 / 50 compared)
V25_PRIOR_C = 0.3    # prior strength, chosen by grouped cross-validation (select_v25.py: best F1)


class SignedLogisticRegression:
    """Logistic regression with sign constraints (red flags >= 0, good signs <= 0) and an
    L2 penalty towards `prior` (all zeros = ordinary L2, as in v2.4)."""

    def __init__(self, C: float = 1.0, columns=None, prior=None):
        self.C = C
        self.columns = list(columns or FLAG_COLUMNS)
        self.prior = np.array([0.0 if prior is None else prior.get(c, 0.0) for c in self.columns])

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        bounds = [(None, 0) if c in PROTECTIVE else (0, None) for c in self.columns] + [(None, None)]

        def loss(beta):
            p = 1 / (1 + np.exp(-(X @ beta[:-1] + beta[-1])))
            ll = -(w * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))).sum() / w.sum()
            return ll + ((beta[:-1] - self.prior) ** 2).sum() / (2 * self.C * len(y))

        res = minimize(loss, np.zeros(X.shape[1] + 1), bounds=bounds, method="L-BFGS-B")
        self.coef_, self.intercept_ = res.x[:-1], res.x[-1]
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(np.asarray(X, float) @ self.coef_ + self.intercept_)))
        return np.c_[1 - p, p]


class HybridModel:
    """Takes a DataFrame with the flag columns (extra columns such as text are ignored)."""

    def __init__(self, C: float = 1.0, columns=None, prior=None):
        self.columns = list(columns or FLAG_COLUMNS)
        self.clf = SignedLogisticRegression(C, self.columns, prior)

    def _cols(self):
        # models saved by v2.4 have no .columns attribute
        return getattr(self, "columns", V24_FLAG_COLUMNS)

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        self.clf.fit(X[self._cols()], y, sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.clf.predict_proba(X[self._cols()])

    def weights(self) -> pd.Series:
        """Each flag's learned weight - the 'what the model learned' chart."""
        return pd.Series(self.clf.coef_, index=self._cols()).sort_values(ascending=False)


def build_v24():
    """The v2.4 design: 33 flags, plain L2 (towards 0), C = 1."""
    return HybridModel(C=1.0, columns=V24_FLAG_COLUMNS, prior=None)


def build_v25():
    """The v2.5 design: 34 flags, L2 towards the expert tier weights."""
    from features import PRIOR_WEIGHTS
    return HybridModel(C=V25_PRIOR_C, columns=FLAG_COLUMNS, prior=PRIOR_WEIGHTS)
