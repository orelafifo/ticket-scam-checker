"""
hybrid_model.py - the v2.4 decision model (Layer 2).

Why it changed (see select_hybrid.py for the evidence):
  v2.3 learned from synthetic listings only, using every word (TF-IDF) plus the red
  flags. It scored ~98% on synthetic data but only 7/20 on an independent set of
  real examples: it had learned the quirks of the templates, not real scams.

What v2.4 does instead:
  1. Uses ONLY the red-flag / good-sign features, not individual words. Words are
     where template quirks hide; the flags are general, explainable signals.
  2. Learns their weights mainly from REAL examples (120 published-case rows,
     weighted x50), with the synthetic listings as background for signals the
     real rows never show (e.g. event checks).
  3. Adds a domain-knowledge constraint: a red flag can only ever RAISE the risk
     and a good sign can only LOWER it. This stops the model learning odd negative
     weights from small data, a common cause of overfitting.
Grouped cross-validation on the real rows (split by source): F1 93% vs 77% for v2.3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from features import FLAG_COLUMNS, PROTECTIVE

REAL_WEIGHT = 50     # chosen by grouped cross-validation (5 / 20 / 50 compared)


class SignedLogisticRegression:
    """Logistic regression with sign constraints: red flags >= 0, good signs <= 0."""

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        bounds = [(None, 0) if c in PROTECTIVE else (0, None) for c in FLAG_COLUMNS] + [(None, None)]

        def loss(beta):
            p = 1 / (1 + np.exp(-(X @ beta[:-1] + beta[-1])))
            ll = -(w * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))).sum() / w.sum()
            return ll + (beta[:-1] ** 2).sum() / (2 * self.C * len(y))

        res = minimize(loss, np.zeros(X.shape[1] + 1), bounds=bounds, method="L-BFGS-B")
        self.coef_, self.intercept_ = res.x[:-1], res.x[-1]
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(np.asarray(X, float) @ self.coef_ + self.intercept_)))
        return np.c_[1 - p, p]


class HybridModel:
    """Takes the same DataFrame as before (text + flag columns) and uses only the flags."""

    def __init__(self, C: float = 1.0):
        self.clf = SignedLogisticRegression(C)

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        self.clf.fit(X[FLAG_COLUMNS], y, sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.clf.predict_proba(X[FLAG_COLUMNS])

    def weights(self) -> pd.Series:
        """Each flag's learned weight - the 'what the model learned' chart."""
        return pd.Series(self.clf.coef_, index=FLAG_COLUMNS).sort_values(ascending=False)
