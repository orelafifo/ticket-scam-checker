"""
compare_v25_v26.py - v2.5 vs v2.6 on the SAME examples.

  python compare_v25_v26.py                                   # development check on blind tests #2 and #3
  python compare_v25_v26.py --test data/blind4_test_set.csv   # blind test #4 (pre-registered, run ONCE)

IMPORTANT: blind tests #2 and #3 are NOT fair tests for v2.6. Blind test #3's misses
(Revolut, TWINT, vague "DM" posts) are what v2.6 was built to fix, so v2.6 will look
better on it. Only blind test #4 (new examples collected before seeing v2.6 results)
is a fair test - see PREREGISTRATION_v26.md.

Approaches (event check and website-age lookup off for all, as in earlier tests):
  v2.5                     saved v2.5 model + frozen v2.5 rules; alert = score >= 0.4 or hard stop
  v2.6 model               v2.6 model; alert = score >= 0.4 or hard stop
  v2.6 as shown in the app also counts "Medium - borderline" and "Medium - not enough
                           information" as an alert (what the user actually sees)
Test rows don't record how the buyer met the seller, so that question counts as unknown.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import features as f26
import rules_v25_frozen as f25
import semantic_flags as sem
from data_utils import metrics, num, real_flags

ap = argparse.ArgumentParser()
ap.add_argument("--test", nargs="*", default=["data/blind2_test_set.csv", "data/blind3_test_set.csv"])
args = ap.parse_args()

v25 = joblib.load("model/scam_model_v25.joblib")
v26 = joblib.load("model/scam_model.joblib")
assert v26.get("version") == "2.6", "run python step2_train_model.py first"
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

all_rows = []
for path in args.test:
    test = pd.read_csv(path)
    y = test.label.values
    X25 = real_flags(test, f25.all_flags)
    p25 = v25["model"].predict_proba(X25)[:, 1]
    a25 = (p25 >= 0.4) | (X25[list(f25.HARD_STOPS)].max(axis=1) == 1).values

    price_status = np.where(test.price.isna(), "not_stated", "stated")
    rows = []
    for i, r in enumerate(test.itertuples()):
        rows.append(real_flags(test.iloc[[i]], f26.all_flags, price_status=price_status[i]).iloc[0])
    X26 = pd.DataFrame(rows)[f26.FLAG_COLUMNS]
    p26 = v26["model"].predict_proba(X26)[:, 1]
    lo, hi = v26["ensemble"].interval(X26)
    hard26 = (X26[list(f26.HARD_STOPS)].max(axis=1) == 1).values
    a26 = (p26 >= 0.4) | hard26
    seller_known = ~(test.account_age_days.isna() & test.followers.isna()).values
    net = np.array([f26.not_enough_info(X26.iloc[i].to_dict(), price_status[i], bool(seller_known[i]), None)
                    for i in range(len(test))])
    shown = a26 | (hi >= 0.4) | net

    name = Path(path).stem
    res = pd.DataFrame({"v2.5": metrics(y, a25.astype(int)), "v2.6 model": metrics(y, a26.astype(int)),
                        "v2.6 as shown in the app": metrics(y, shown.astype(int))}).T
    res.insert(0, "test_set", name)
    all_rows.append(res)
    per = test[["id", "label", "platform"]].copy()
    per["v25_score"], per["v26_score"] = p25.round(2), p26.round(2)
    per["v26_range"] = [f"{a:.2f}-{b:.2f}" for a, b in zip(lo, hi)]
    per["v25_alert"], per["v26_alert"], per["v26_shown_medium_plus"] = a25.astype(int), a26.astype(int), shown.astype(int)
    per.to_csv(OUT / f"v25_vs_v26_{name}_predictions.csv", index=False)
sem.save_cache()

res = pd.concat(all_rows)
res.to_csv(OUT / "v25_vs_v26.csv")
fmt = {"recall": "{:.0%}".format, "precision": "{:.0%}".format, "F1": "{:.0%}".format,
       "false_alarm_rate": "{:.0%}".format}
print(res.to_string(formatters=fmt))

fig, axes = plt.subplots(1, len(args.test), figsize=(5.2 * len(args.test), 4), squeeze=False)
for ax, (name, g) in zip(axes[0], res.groupby("test_set", sort=False)):
    xs = np.arange(len(g))
    ax.bar(xs - 0.2, g.recall.astype(float), 0.38, color="#2a78d6", label="Scams caught")
    ax.bar(xs + 0.2, g.false_alarm_rate.astype(float), 0.38, color="#eb6834", label="False alarms")
    for x, (a, b) in enumerate(zip(g.recall, g.false_alarm_rate)):
        ax.text(x - 0.2, a + 0.02, f"{a:.0%}", ha="center", fontsize=8)
        ax.text(x + 0.2, b + 0.02, f"{b:.0%}", ha="center", fontsize=8)
    ax.set_xticks(xs, [i.replace(" as shown in the app", "\n(app view)") for i in g.index], fontsize=8)
    ax.set_ylim(0, 1.15); ax.set_title(name, loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig(OUT / "v25_vs_v26.png", dpi=200); plt.close()
print("Saved outputs/v25_vs_v26.csv, v25_vs_v26.png and per-example predictions.")
