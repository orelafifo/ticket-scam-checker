"""
select_v27.py - v2.7 Upgrade 1: is the MiniLM meaning score worth keeping?

Follows PREREGISTRATION_v27.md. Uses ONLY training rows (291 real + synthetic background);
blind test #5 does not exist yet and is never touched here.

  Design A   v2.6 design, retrained on all real rows (no meaning score)
  Design B   A + meaning_score, at C = 0.01 / 0.1 / 1 (strength of the meaning model)

Method (same as select_v25 / select_v26): source-grouped 5-fold cross-validation, repeated
5 times, real rows weight x50, threshold 0.4 + hard-stop rules. Inside every fold the
meaning model is trained on that fold's training rows only, and those rows get
out-of-fold meaning scores, so no row is ever scored by a model that saw its label.

Pre-registered rule: keep B (best C) only if its F1 >= A's AND its false-alarm rate is no
more than 2 points higher. Otherwise v2.7 ships design A.

Extra (fair, because v2.6 never saw them): how the FROZEN v2.6 model does on the 51 new rows.

Run:  python select_v27.py      Output: outputs/select_v27.md, outputs/select_v27.csv
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import features as f26
import semantic_flags as sem
from data_utils import REAL_FILES_V27, load_real, metrics, real_flags, synth_flags
from hybrid_model import REAL_WEIGHT, build_v27
from meaning_model import MeaningModel, oof_scores

THRESHOLD = 0.4
C_VALUES = [0.01, 0.1, 1.0]
REPEATS = 5

if not sem.available():
    raise SystemExit("MiniLM not available - install sentence-transformers and run once with internet.")

real = load_real(REAL_FILES_V27)
yR, gR, tR = real.label.values, real.group.values, real.text.astype(str).values
syn = pd.read_csv("data/listings.csv")
yS, tS = syn.label.values, syn.text.astype(str).values
print(f"Training rows: {len(real)} real ({int(yR.sum())} scam / {int((1 - yR).sum())} genuine) "
      f"+ {len(syn)} synthetic background")
print("Building red-flag tables and MiniLM embeddings (first run takes a few minutes)...")
S = synth_flags(syn, f26.all_flags)
R = real_flags(real, f26.all_flags)
sem.save_cache()
hard_all = (R[list(f26.HARD_STOPS)].max(axis=1) == 1).values


def fit_predict(train, val, C):
    """Train on synthetic + real[train], return scores for real[val]. C=None = design A."""
    S_, Rtr, Rva = S.copy(), R.iloc[train].copy(), R.iloc[val].copy()
    if C is not None:
        Rtr["meaning_score"] = oof_scores(tR[train], yR[train], gR[train], C)
        m = MeaningModel(C).fit(tR[train], yR[train])
        Rva["meaning_score"] = m.score(tR[val])
        S_["meaning_score"] = m.score(tS)
    model = build_v27(with_meaning=C is not None).fit(
        pd.concat([S_, Rtr], ignore_index=True), np.r_[yS, yR[train]],
        np.r_[np.ones(len(S_)), np.full(len(train), REAL_WEIGHT)])
    return model.predict_proba(Rva)[:, 1]


def cross_validate(C):
    out, oof = [], np.zeros((REPEATS, len(R)))
    for repeat in range(REPEATS):
        rng = np.random.RandomState(repeat)
        groups = pd.Series(gR).map({g: rng.rand() for g in pd.unique(gR)}).rank(method="dense").values
        pred = np.zeros(len(R), int)
        for train, val in GroupKFold(n_splits=5).split(R, yR, groups):
            p = fit_predict(train, val, C)
            oof[repeat, val] = p
            pred[val] = ((p >= THRESHOLD) | hard_all[val]).astype(int)
        out.append(metrics(yR, pred))
    df = pd.DataFrame(out)
    return {k: df[k].mean() for k in ["F1", "recall", "precision", "false_alarm_rate"]}, oof.mean(axis=0)


rows, oofs = [], {}
res, oofs["A"] = cross_validate(None)
rows.append(dict(design="A: v2.6 design, all real rows", C="-", **res))
print("Design A done")
for C in C_VALUES:
    res, oofs[C] = cross_validate(C)
    rows.append(dict(design=f"B: A + meaning score", C=C, **res))
    print(f"Design B, C={C} done")
sem.save_cache()
res = pd.DataFrame(rows)

A = res.iloc[0]
B = res.iloc[1:].sort_values(["F1", "false_alarm_rate"], ascending=[False, True]).iloc[0]
keep = (B.F1 >= A.F1) and (B.false_alarm_rate <= A.false_alarm_rate + 0.02)

# ---------------------------------------------------------------- frozen v2.6 on the 51 new rows
new = real.file.values == "v27_new_real_rows"
v26 = joblib.load("model/scam_model_v26.joblib")
p26 = v26["model"].predict_proba(R[new][f26.FLAG_COLUMNS])[:, 1]
pred26 = ((p26 >= THRESHOLD) | hard_all[new]).astype(int)
m26 = metrics(yR[new], pred26)
best_oof = oofs[B.C] if keep else oofs["A"]
predCV = ((best_oof[new] >= THRESHOLD) | hard_all[new]).astype(int)
mCV = metrics(yR[new], predCV)

# ---------------------------------------------------------------- paraphrase sentences: does the meaning score react?
para = pd.read_csv("data/paraphrase_stress.csv")
mm = MeaningModel(B.C).fit(tR, yR)
para["meaning_score"] = mm.score(para.text.astype(str).values)
sem.save_cache()
pos, neg = para[para.should_fire == 1], para[para.should_fire == 0]

# ---------------------------------------------------------------- report
pct = lambda v: f"{v:.0%}"
lines = ["# v2.7 Upgrade 1: does the meaning score help?", "",
         f"{len(real)} real rows ({int(yR.sum())} scam, {int((1 - yR).sum())} genuine) + {len(syn)} synthetic. "
         f"Source-grouped 5-fold cross-validation, repeated {REPEATS} times. Threshold 0.4 + hard stops.", "",
         "| Design | Meaning-model C | F1 | Scams caught | Precision | False alarms |", "|---|---|---|---|---|---|"]
for r in res.itertuples():
    lines.append(f"| {r.design} | {r.C} | {r.F1:.1%} | {pct(r.recall)} | {pct(r.precision)} | {pct(r.false_alarm_rate)} |")
lines += ["", f"Best B: C = {B.C}. Pre-registered rule: keep B only if F1 >= A ({B.F1:.1%} vs {A.F1:.1%}) "
          f"and false alarms are no more than 2 points higher ({pct(B.false_alarm_rate)} vs {pct(A.false_alarm_rate)}).",
          "", f"**Decision: {'KEEP the meaning score (design B)' if keep else 'DO NOT keep it - v2.7 ships design A'}.**", "",
          "## The 51 new UK music rows (v2.6 never saw them)", "",
          "| Model | Scams caught | False alarms | F1 |", "|---|---|---|---|",
          f"| v2.6 frozen | {m26['caught']} | {m26['false_alarms']} | {m26['F1']:.0%} |",
          f"| v2.7 chosen design (cross-validated) | {mCV['caught']} | {mCV['false_alarms']} | {mCV['F1']:.0%} |", "",
          "## Paraphrase sentences (AI-written): average meaning score", "",
          f"- Reworded red-flag sentences: {pos.meaning_score.mean():.2f} ({(pos.meaning_score >= 0.5).sum()}/{len(pos)} at 0.5 or above)",
          f"- Ordinary listing sentences: {neg.meaning_score.mean():.2f} ({(neg.meaning_score >= 0.5).sum()}/{len(neg)} at 0.5 or above)",
          "", "Limits: most new scam rows were reconstructed from descriptions, not quoted; the 51-row "
          "check is cross-validated, not a blind test. Blind test #5 is the real test."]
Path("outputs").mkdir(exist_ok=True)
res.to_csv("outputs/select_v27.csv", index=False)
Path("outputs/select_v27.md").write_text("\n".join(lines), encoding="utf-8")
print("\n" + "\n".join(lines))
