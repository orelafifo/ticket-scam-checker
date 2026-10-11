"""
select_v25.py - choose the v2.5 prior strength, and check v2.5 against the v2.4 design,
WITHOUT touching any blind test set.

Same method as select_hybrid.py / compare_regularisation.py:
  * 120 real rows (data/real_test_set.csv + data/heldout_test_set.csv, weight x50)
    + synthetic background (data/listings.csv, weight x1)
  * source-grouped 5-fold cross-validation, repeated 5 times
  * threshold 0.4 + hard-stop rules

Designs compared
  v2.4            33 flags, L2 towards 0 (C=1)                        - the current model
  v2.4 + price    34 flags (adds very_cheap), L2 towards 0             - isolates the new flag
  v2.5, C=...     34 flags, L2 towards the expert tier weights         - the new idea, several strengths
                  (small C = weights stay close to the expert tiers; large C = data decides more)

Note: the real rows have no event-check data, so the event flags' weights come from the
expert tiers + synthetic data and can't be judged by this cross-validation. That is one
reason a new blind test is still needed.

Run:  python select_v25.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from channels import channel_info
from features import FLAG_COLUMNS, HARD_STOPS, PRIOR_WEIGHTS, V24_FLAG_COLUMNS, all_flags
from hybrid_model import REAL_WEIGHT, HybridModel

THRESHOLD = 0.4


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
real["group"] = real["source"].fillna("constructed").where(real.label == 1, real["id"])
R, yR = real_flags(real), real.label.values
syn = pd.read_csv("data/listings.csv")
S, yS = synth_flags(syn), syn.label.values

DESIGNS = [("v2.4 (current)", lambda: HybridModel(1.0, V24_FLAG_COLUMNS, None)),
           ("v2.4 + very_cheap", lambda: HybridModel(1.0, FLAG_COLUMNS, None))]
for C in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
    DESIGNS.append((f"v2.5 priors, C={C}", lambda C=C: HybridModel(C, FLAG_COLUMNS, PRIOR_WEIGHTS)))


def cross_validate(make):
    scores = []
    for repeat in range(5):
        rng = np.random.RandomState(repeat)
        groups = real["group"].map({g: rng.rand() for g in real["group"].unique()}).rank(method="dense")
        pred = np.zeros(len(real), int)
        for train, val in GroupKFold(n_splits=5).split(R, yR, groups):
            X = pd.concat([S, R.iloc[train]], ignore_index=True)
            y = np.r_[yS, yR[train]]
            w = np.r_[np.ones(len(S)), np.full(len(train), REAL_WEIGHT)]
            m = make().fit(X, y, w)
            p = m.predict_proba(R.iloc[val])[:, 1]
            hard = (R.iloc[val][list(HARD_STOPS)].max(axis=1) == 1).values
            pred[val] = ((p >= THRESHOLD) | hard).astype(int)
        tp = ((pred == 1) & (yR == 1)).sum(); fp = ((pred == 1) & (yR == 0)).sum()
        fn = ((pred == 0) & (yR == 1)).sum()
        scores.append([2 * tp / (2 * tp + fp + fn), tp / (tp + fn), fp / (yR == 0).sum()])
    return np.mean(scores, axis=0)


rows = []
for name, make in DESIGNS:
    f1, rec, fa = cross_validate(make)
    rows.append(dict(design=name, F1=f1, recall=rec, false_alarm_rate=fa))
res = pd.DataFrame(rows)
Path("outputs").mkdir(exist_ok=True)
res.to_csv("outputs/select_v25.csv", index=False)
print("Source-grouped 5-fold cross-validation, repeated 5 times (120 real rows + synthetic background)\n")
print(res.to_string(index=False, formatters={"F1": "{:.1%}".format, "recall": "{:.0%}".format,
                                             "false_alarm_rate": "{:.0%}".format}))

# Weights learned by each design on ALL the training data, side by side
full_X = pd.concat([S, R], ignore_index=True)
full_y = np.r_[yS, yR]
full_w = np.r_[np.ones(len(S)), np.full(len(R), REAL_WEIGHT)]
v24 = HybridModel(1.0, V24_FLAG_COLUMNS, None).fit(full_X, full_y, full_w).weights()
best = res[res.design.str.startswith("v2.5")].sort_values("F1", ascending=False).iloc[0]
C_best = float(best.design.split("C=")[1])
v25 = HybridModel(C_best, FLAG_COLUMNS, PRIOR_WEIGHTS).fit(full_X, full_y, full_w).weights()
w = pd.DataFrame({"expert_prior": pd.Series(PRIOR_WEIGHTS), "v2.4_weight": v24, f"v2.5_weight(C={C_best})": v25})
w.to_csv("outputs/weights_v24_vs_v25.csv")
print(f"\nBest v2.5 strength by F1: C={C_best}. Weights compared in outputs/weights_v24_vs_v25.csv")
print(w.sort_values("expert_prior", ascending=False).round(2).to_string())
