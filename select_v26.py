"""
select_v26.py - choose the MiniLM similarity threshold and compare v2.6 with v2.5,
WITHOUT touching any future blind test.

Same method as select_v25.py:
  * 120 real rows (published cases, weight x50) + synthetic background (weight x1)
  * source-grouped 5-fold cross-validation, repeated 5 times
  * threshold 0.4 + hard-stop rules

Designs compared
  v2.5 (frozen)              the current model, rebuilt exactly (rules_v25_frozen.py)
  v2.6 rules only            + text normalisation + new phrases (DM, Revolut, TWINT...), no AI
  v2.6 + MiniLM, sim=...     + the AI meaning-matcher at several similarity thresholds

Also reports, on the same 120 rows, how often the new safety net ("not enough information")
and the borderline band would change what the user sees - so its cost in false alarms is known.

Run:  python select_v26.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import features as f26
import rules_v25_frozen as f25
import semantic_flags as sem
from data_utils import load_real, metrics, real_flags, synth_flags
from hybrid_model import REAL_WEIGHT, HybridModel, build_v25, build_v26

THRESHOLD = 0.4
SIMS = [0.55, 0.60, 0.65, 0.70]

real = load_real()
yR = real.label.values
syn = pd.read_csv("data/listings.csv")
yS = syn.label.values


def tables(flag_fn, **kw):
    return synth_flags(syn, flag_fn, **kw), real_flags(real, flag_fn, **kw)


def cross_validate(S, R, make, hard_stops):
    out, oof = [], np.zeros((5, len(R)))
    for repeat in range(5):
        rng = np.random.RandomState(repeat)
        groups = real["group"].map({g: rng.rand() for g in real["group"].unique()}).rank(method="dense")
        pred = np.zeros(len(R), int)
        for train, val in GroupKFold(n_splits=5).split(R, yR, groups):
            m = make().fit(pd.concat([S, R.iloc[train]], ignore_index=True), np.r_[yS, yR[train]],
                           np.r_[np.ones(len(S)), np.full(len(train), REAL_WEIGHT)])
            p = m.predict_proba(R.iloc[val])[:, 1]
            oof[repeat, val] = p
            hard = (R.iloc[val][list(hard_stops)].max(axis=1) == 1).values
            pred[val] = ((p >= THRESHOLD) | hard).astype(int)
        out.append(metrics(yR, pred))
    df = pd.DataFrame(out)
    return {k: df[k].mean() for k in ["F1", "recall", "precision", "false_alarm_rate"]}, oof.mean(axis=0)


rows, store = [], {}
print("Building flag tables (MiniLM embeds each sentence once, then it's cached)...")
S, R = tables(f25.all_flags)
res, _ = cross_validate(S, R, build_v25, f25.HARD_STOPS)
rows.append(dict(design="v2.5 (frozen)", **res))

S, R = tables(f26.all_flags, semantic=False)
res, _ = cross_validate(S, R, build_v26, f26.HARD_STOPS)
rows.append(dict(design="v2.6 rules only (no AI)", **res))
store["rules"] = (S, R)

if not sem.available():
    raise SystemExit("MiniLM not available - install sentence-transformers and run once with internet.")
for t in SIMS:
    sem.SIM_THRESHOLD = t
    S, R = tables(f26.all_flags, semantic=True)
    res, oof = cross_validate(S, R, build_v26, f26.HARD_STOPS)
    rows.append(dict(design=f"v2.6 + MiniLM, sim>={t:.2f}", **res))
    store[t] = (S, R, oof)
sem.save_cache()

res = pd.DataFrame(rows)
Path("outputs").mkdir(exist_ok=True)
res.to_csv("outputs/select_v26.csv", index=False)
print("\nSource-grouped 5-fold cross-validation, repeated 5 times (120 real rows + synthetic background)\n")
print(res.to_string(index=False, formatters={"F1": "{:.1%}".format, "recall": "{:.0%}".format,
                                             "precision": "{:.0%}".format, "false_alarm_rate": "{:.0%}".format}))

ai = res[res.design.str.contains("MiniLM")].copy()
ai["sim"] = SIMS
best = ai.sort_values(["F1", "sim"], ascending=[False, False]).iloc[0]
print(f"\nChosen similarity threshold: {best.sim:.2f} (best F1; ties -> the stricter threshold). "
      f"Set semantic_flags.SIM_THRESHOLD to this value.")

# How often does the MiniLM matcher add a flag the rules missed? (the AI's own contribution)
S_r, R_r = store["rules"]
S_a, R_a, oof = store[best.sim]
added = (R_a[R_r.columns] - R_r).clip(lower=0)
print("\nFlags added by MiniLM on the 120 real rows (rules missed them):")
print(pd.DataFrame({"added_on_scams": added[yR == 1].sum(), "added_on_genuine": added[yR == 0].sum()})
      .query("added_on_scams + added_on_genuine > 0").to_string())

# Safety net + borderline: what would the user see? (cross-validated scores)
flags = R_a
seller_known = ~(real.account_age_days.isna() & real.followers.isna())
price_status = np.where(real.price.isna(), "not_stated", "stated")
net = np.array([f26.not_enough_info(flags.iloc[i].to_dict(), price_status[i], bool(seller_known.iloc[i]), None)
                for i in range(len(real))])
hard = (flags[list(f26.HARD_STOPS)].max(axis=1) == 1).values
alert = (oof >= THRESHOLD) | hard
shown_medium_plus = alert | net
print("\nWhat the user would see (cross-validated scores, chosen design):")
print(f"  Low-risk results turned into 'Medium - not enough information' by the safety net: "
      f"{int((net & ~alert & (yR == 1)).sum())} scams, {int((net & ~alert & (yR == 0)).sum())} genuine")
print("  Alert or 'not enough information':", metrics(yR, shown_medium_plus.astype(int)))
