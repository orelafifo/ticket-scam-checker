"""
select_hybrid.py - choose a less overfit hybrid design WITHOUT touching any future test set.

Problem found: the hybrid's ML combiner, trained only on synthetic listings, learned
weights that don't transfer to real posts (dev set 19/40 -> 35/40 after tuning,
then 7/20 on an independent held-out set).

Method: the 120 real-world rows we have already seen (80 development + 40 first
held-out, both now "used") are pooled and split by SOURCE, so examples quoted
from the same article never sit on both sides of a split (grouped 5-fold
cross-validation, repeated). Each candidate design is scored the same way and the
winner is picked by mean F1. The final judgement then comes from a NEW blind set.

Candidates
  A  current hybrid: TF-IDF words + flags, trained on synthetic only
  B  flags only, trained on synthetic only
  C  flags only, trained on real rows (cross-validated)
  D  flags only, trained on synthetic + real (real rows weighted x5)
  E  D, but red-flag weights forced to be >= 0 (a red flag can only ever raise
     the risk) and good-sign weights <= 0 - domain knowledge as a constraint
  F  E, with a 'rules floor': if the rules-only decision says scam, so does the hybrid
  G  flags only, trained on real rows only, with the same sign constraint as E
  H  G with the rules floor
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from channels import channel_info
from features import FLAG_COLUMNS, HARD_STOPS, PROTECTIVE, all_flags

WEAK = {"no_seat_details", "high_demand"}
MEANINGFUL = [c for c in FLAG_COLUMNS if c not in WEAK and c not in PROTECTIVE]


# ------------------------------------------------------------------ data
def num(v, d):
    return d if pd.isna(v) or v == "" else float(v)


def real_flags(df):
    rows = []
    for r in df.itertuples():
        p, f = num(r.price, 0), num(r.face_value, 0)
        ch = channel_info(r.platform, r.link if isinstance(r.link, str) else "", check_age=False)
        rows.append(all_flags(r.text, num(r.account_age_days, 365), num(r.followers, 200),
                              p / f if p and f else 1.0, int(r.has_seat_details), int(r.sudden_seller), None, ch))
    return pd.DataFrame(rows)


def tri(v):
    return None if v == -1 else bool(v)


def synth_flags(df):
    return pd.DataFrame([all_flags(
        r.text, r.account_age_days, r.followers, r.price_ratio, r.has_seat_details, r.sudden_seller,
        dict(found=tri(r.event_found), details_match=tri(r.event_details_match),
             high_demand=bool(r.event_high_demand), before_on_sale=bool(r.event_before_on_sale),
             app_only=bool(r.event_app_only)),
        dict(category=None if r.ch_category == "none" else r.ch_category, lookalike=bool(r.ch_lookalike),
             new_domain=bool(r.ch_new_domain), insecure_or_short=bool(r.ch_insecure_or_short)))
        for r in df.itertuples()])


dev = pd.read_csv("data/real_test_set.csv")
ho = pd.read_csv("data/heldout_test_set.csv")
real = pd.concat([dev, ho], ignore_index=True)
real["group"] = real["source"].fillna("constructed").where(real.label == 1, real["id"])  # genuine rows: own group
R = real_flags(real)
yR = real.label.values
syn = pd.read_csv("data/listings.csv")
S = synth_flags(syn)
yS = syn.label.values


# ------------------------------------------------------------------ constrained logistic regression
class SignedLR:
    """Logistic regression where red flags get weight >= 0 and good signs <= 0."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y, w=None):
        X = np.asarray(X, float); w = np.ones(len(y)) if w is None else w
        n = X.shape[1]
        bounds = [(None, 0) if c in PROTECTIVE else (0, None) for c in FLAG_COLUMNS] + [(None, None)]

        def loss(beta):
            z = X @ beta[:-1] + beta[-1]
            p = 1 / (1 + np.exp(-z))
            eps = 1e-9
            ll = -(w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))).sum() / w.sum()
            return ll + (beta[:-1] ** 2).sum() / (2 * self.C * len(y))

        res = minimize(loss, np.zeros(n + 1), bounds=bounds, method="L-BFGS-B")
        self.coef_, self.intercept_ = res.x[:-1], res.x[-1]
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(np.asarray(X, float) @ self.coef_ + self.intercept_)))
        return np.c_[1 - p, p]


def rules_decision(F):
    return ((F[list(HARD_STOPS)].max(axis=1) == 1) | (F[MEANINGFUL].sum(axis=1) >= 2)).astype(int).values


def f1(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    return 2 * tp / (2 * tp + fp + fn), tp / (tp + fn), fp / max((y == 0).sum(), 1)


# ------------------------------------------------------------------ candidates
def run(name, fold_train, fold_test, thr=0.4):
    """returns predictions for the test fold"""
    tr, te = real.iloc[fold_train], real.iloc[fold_test]
    Rtr, Rte = R.iloc[fold_train], R.iloc[fold_test]
    hard = (Rte[list(HARD_STOPS)].max(axis=1) == 1).values
    if name == "A":
        m = Pipeline([("f", ColumnTransformer([("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2), "text"),
                                               ("x", "passthrough", FLAG_COLUMNS)])),
                      ("c", LogisticRegression(max_iter=1000, class_weight="balanced"))])
        m.fit(pd.concat([syn[["text"]], S], axis=1), yS)
        p = m.predict_proba(pd.concat([te[["text"]].reset_index(drop=True), Rte.reset_index(drop=True)], axis=1))[:, 1]
    elif name == "B":
        p = LogisticRegression(max_iter=1000, class_weight="balanced").fit(S, yS).predict_proba(Rte)[:, 1]
    elif name == "C":
        p = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Rtr, tr.label).predict_proba(Rte)[:, 1]
    elif name in ("G", "H"):
        p = SignedLR(C=1.0).fit(Rtr, tr.label.values).predict_proba(Rte)[:, 1]
    else:
        X = pd.concat([S, Rtr], ignore_index=True); y = np.r_[yS, tr.label.values]
        w = np.r_[np.ones(len(S)), np.full(len(Rtr), 5.0)]
        if name == "D":
            p = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y, sample_weight=w).predict_proba(Rte)[:, 1]
        else:
            p = SignedLR(C=1.0).fit(X, y, w).predict_proba(Rte)[:, 1]
    pred = ((p >= thr) | hard).astype(int)
    if name in ("F", "H"):
        pred = np.maximum(pred, rules_decision(Rte))
    return pred


results = {}
for name in "ABCDEFGH":
    scores = []
    for rep in range(5):  # repeat with shuffled group order
        rng = np.random.RandomState(rep)
        groups = real["group"].map({g: rng.rand() for g in real["group"].unique()})
        pred = np.zeros(len(real), int)
        for trn, tst in GroupKFold(n_splits=5).split(R, yR, groups.rank(method="dense")):
            pred[tst] = run(name, trn, tst)
        scores.append(f1(yR, pred))
    results[name] = np.mean(scores, axis=0)

rules = f1(yR, rules_decision(R))
print("Grouped 5-fold CV on 120 already-seen real rows (mean of 5 repeats)")
print(f"{'design':6s} {'F1':>6s} {'recall':>7s} {'false-alarm rate':>17s}")
print(f"{'rules':6s} {rules[0]:6.0%} {rules[1]:7.0%} {rules[2]:17.0%}   (no training, reference)")
for k, (a, b, c) in results.items():
    print(f"{k:6s} {a:6.0%} {b:7.0%} {c:17.0%}")


# ------------------------------------------------------------------ threshold for the chosen design (G)
print("\nThreshold check for design G (same grouped CV)")
for thr in [0.3, 0.4, 0.5, 0.6]:
    sc = []
    for rep in range(5):
        rng = np.random.RandomState(rep)
        groups = real["group"].map({g: rng.rand() for g in real["group"].unique()})
        pred = np.zeros(len(real), int)
        for trn, tst in GroupKFold(n_splits=5).split(R, yR, groups.rank(method="dense")):
            pred[tst] = run("G", trn, tst, thr)
        sc.append(f1(yR, pred))
    a, b, c = np.mean(sc, axis=0)
    print(f"  threshold {thr}: F1 {a:.0%}, recall {b:.0%}, false alarms {c:.0%}")
