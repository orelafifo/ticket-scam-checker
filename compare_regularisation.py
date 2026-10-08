"""
compare_regularisation.py - does a different penalty (L1 / Lasso or L2 / Ridge) or
strength improve the v2.4 model?

Method (same as select_hybrid.py, so results are comparable):
  * Data: the 120 real-world rows (data/real_test_set.csv + data/heldout_test_set.csv,
    weighted x50) plus the synthetic background rows (data/listings.csv, weight x1).
  * Model: the v2.4 design - flag features only, red flags >= 0, good signs <= 0.
  * Only the PENALTY changes:
        L2 (Ridge):  penalty = sum(w^2) / (2 * C * n)   -> shrinks all weights a little
        L1 (Lasso):  penalty = sum(|w|) / (C * n)       -> can push weights to exactly 0
    C = strength setting: SMALLER C = STRONGER penalty.
  * Scored with source-grouped 5-fold cross-validation, repeated 5 times
    (examples from the same article never appear in both training and validation).
  * The blind test set (data/blind2_test_set.csv) is NOT used, so it stays a fair final test.

Run:  python compare_regularisation.py
Output: printed table + outputs/regularisation_comparison.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

from channels import channel_info
from features import FLAG_COLUMNS, HARD_STOPS, PROTECTIVE, all_flags
from hybrid_model import REAL_WEIGHT

THRESHOLD = 0.4
SETTINGS = [("L2", 0.01), ("L2", 0.1), ("L2", 1.0), ("L2", 10.0),
            ("L1", 0.01), ("L1", 0.1), ("L1", 1.0), ("L1", 10.0)]
CURRENT = ("L2", 1.0)   # what v2.4 uses


# ---------------------------------------------------------------- data (same as step2 / select_hybrid)
def num(v, d):
    return d if pd.isna(v) or v == "" else float(v)


def tri(v):
    return None if v == -1 else bool(v)


def real_flags(df):
    rows = []
    for r in df.itertuples():
        p, f = num(r.price, 0), num(r.face_value, 0)
        ch = channel_info(r.platform, r.link if isinstance(r.link, str) else "", check_age=False)
        rows.append(all_flags(r.text, num(r.account_age_days, 365), num(r.followers, 200),
                              p / f if p and f else 1.0, int(r.has_seat_details), int(r.sudden_seller), None, ch))
    return pd.DataFrame(rows)


def synth_flags(df):
    return pd.DataFrame([all_flags(
        r.text, r.account_age_days, r.followers, r.price_ratio, r.has_seat_details, r.sudden_seller,
        dict(found=tri(r.event_found), details_match=tri(r.event_details_match),
             high_demand=bool(r.event_high_demand), before_on_sale=bool(r.event_before_on_sale),
             app_only=bool(r.event_app_only)),
        dict(category=None if r.ch_category == "none" else r.ch_category, lookalike=bool(r.ch_lookalike),
             new_domain=bool(r.ch_new_domain), insecure_or_short=bool(r.ch_insecure_or_short)))
        for r in df.itertuples()])


real = pd.concat([pd.read_csv("data/real_test_set.csv"), pd.read_csv("data/heldout_test_set.csv")],
                 ignore_index=True)
real["group"] = real["source"].fillna("constructed").where(real.label == 1, real["id"])  # split by source
R, yR = real_flags(real), real.label.values
syn = pd.read_csv("data/listings.csv")
S, yS = synth_flags(syn), syn.label.values

SIGN = np.array([-1 if c in PROTECTIVE else 1 for c in FLAG_COLUMNS])
BOUNDS = [(None, 0) if c in PROTECTIVE else (0, None) for c in FLAG_COLUMNS] + [(None, None)]


# ---------------------------------------------------------------- the model with a choice of penalty
def fit(X, y, w, penalty, C):
    X = np.asarray(X, float)

    def loss(b):
        p = 1 / (1 + np.exp(-(X @ b[:-1] + b[-1])))
        log_loss = -(w * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))).sum() / w.sum()
        if penalty == "L2":
            reg = (b[:-1] ** 2).sum() / (2 * C * len(y))
        else:  # L1: |w| equals sign * w here, because the sign of every weight is fixed
            reg = (SIGN * b[:-1]).sum() / (C * len(y))
        return log_loss + reg

    return minimize(loss, np.zeros(X.shape[1] + 1), bounds=BOUNDS, method="L-BFGS-B").x


def cross_validate(penalty, C):
    scores = []
    for repeat in range(5):
        rng = np.random.RandomState(repeat)
        groups = real["group"].map({g: rng.rand() for g in real["group"].unique()}).rank(method="dense")
        pred = np.zeros(len(real), int)
        for train, val in GroupKFold(n_splits=5).split(R, yR, groups):
            X = pd.concat([S, R.iloc[train]], ignore_index=True)
            y = np.r_[yS, yR[train]]
            w = np.r_[np.ones(len(S)), np.full(len(train), REAL_WEIGHT)]
            beta = fit(X, y, w, penalty, C)
            Xv = np.asarray(R.iloc[val], float)
            p = 1 / (1 + np.exp(-(Xv @ beta[:-1] + beta[-1])))
            hard = (R.iloc[val][list(HARD_STOPS)].max(axis=1) == 1).values
            pred[val] = ((p >= THRESHOLD) | hard).astype(int)
        tp = ((pred == 1) & (yR == 1)).sum(); fp = ((pred == 1) & (yR == 0)).sum()
        fn = ((pred == 0) & (yR == 1)).sum()
        scores.append([2 * tp / (2 * tp + fp + fn), tp / (tp + fn), fp / (yR == 0).sum()])
    beta = fit(pd.concat([S, R], ignore_index=True), np.r_[yS, yR],
               np.r_[np.ones(len(S)), np.full(len(R), REAL_WEIGHT)], penalty, C)
    zero = int((np.abs(beta[:-1]) < 1e-6).sum())  # how many flags the penalty switched off
    f1, rec, fa = np.mean(scores, axis=0)
    return dict(penalty=penalty, C=C, F1=f1, recall=rec, false_alarm_rate=fa, flags_switched_off=zero)


rows = [cross_validate(p, c) for p, c in SETTINGS]
res = pd.DataFrame(rows)
res["note"] = ["<- current v2.4" if (r.penalty, r.C) == CURRENT else "" for r in res.itertuples()]
Path("outputs").mkdir(exist_ok=True)
res.to_csv("outputs/regularisation_comparison.csv", index=False)

print("Source-grouped 5-fold cross-validation, repeated 5 times (120 real rows + synthetic background)\n")
print(res.to_string(index=False, formatters={"F1": "{:.1%}".format, "recall": "{:.0%}".format,
                                             "false_alarm_rate": "{:.0%}".format}))
best = res.loc[res.F1.idxmax()]
cur = res[(res.penalty == CURRENT[0]) & (res.C == CURRENT[1])].iloc[0]
print(f"\nBest: {best.penalty} C={best.C} (F1 {best.F1:.1%}) vs current {CURRENT[0]} C={CURRENT[1]} "
      f"(F1 {cur.F1:.1%}): difference {100 * (best.F1 - cur.F1):.1f} points.")
print("Saved outputs/regularisation_comparison.csv")
