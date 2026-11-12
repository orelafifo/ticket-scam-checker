"""
meaning_model.py - v2.7 Upgrade 1: the MiniLM "meaning score".

What it does, in plain words:
  * MiniLM (the same small local model as semantic_flags.py) turns the WHOLE listing or
    conversation into 384 numbers that capture what it means.
  * A small logistic regression learns, from the real labelled examples only, which
    meanings go with scams. Its output is one number from 0 to 1: the "meaning score".
  * That one number is added to the decision model as ONE extra feature. The decision
    model still works on red flags with expert-tier priors; words are NOT added back
    (the v2.4 overfitting lesson). The meaning score can only raise risk (sign >= 0),
    and its starting weight is the moderate tier (1.5).

No leakage: when the decision model is trained, each real row's meaning score comes
from a meaning model that did NOT see that row (out-of-fold scores, 5 groups).
Everything runs on this computer; nothing is sent to an outside service.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

import semantic_flags as sem
from normalise import normalise


def embed_texts(texts) -> np.ndarray:
    return sem.embed([normalise(str(t)) for t in texts])


class MeaningModel:
    def __init__(self, C: float = 0.1):
        self.C = C

    def fit(self, texts, y):
        self.clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000)
        self.clf.fit(embed_texts(texts), np.asarray(y))
        return self

    def score(self, texts) -> np.ndarray:
        return self.clf.predict_proba(embed_texts(texts))[:, 1]


def oof_scores(texts, y, groups, C: float, n_splits: int = 5) -> np.ndarray:
    """Meaning score for each training row from a model that never saw that row's group."""
    texts, y, groups = np.asarray(texts, dtype=object), np.asarray(y), np.asarray(groups)
    out = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=n_splits).split(texts, y, groups):
        out[va] = MeaningModel(C).fit(texts[tr], y[tr]).score(texts[va])
    return out
