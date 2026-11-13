"""
step2_train_v27.py - train the v2.7 decision model (the one the app uses).

Same as step2_train_model.py (v2.6), plus Upgrade 1:
  * trained on all real rows in REAL_FILES_V27 (294 as of 13 Nov) + synthetic background
  * ONE extra feature, meaning_score, from the MiniLM meaning model (meaning_model.py),
    C = 1.0 as chosen by select_v27.py under the pre-registered rule.
    Real rows get out-of-fold meaning scores, so the decision model never learns from a
    score that already saw the row's label.
  * the 30-model ensemble (uncertainty / "borderline") is rebuilt the same way as v2.6.

v2.6 is kept frozen in model/scam_model_v26.joblib (and git tag v2.6).
Run:  python step2_train_v27.py      Output: model/scam_model.joblib (v2.7)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import semantic_flags as sem
from data_utils import REAL_FILES_V27, load_real, real_flags, synth_flags
from features import all_flags
from hybrid_model import REAL_WEIGHT, Ensemble, build_v27
from meaning_model import MeaningModel, oof_scores

THRESHOLD = 0.4
MEANING_C = 1.0      # chosen by select_v27.py
N_ENSEMBLE = 30

if not sem.available():
    raise SystemExit("MiniLM not available - install sentence-transformers and run once with internet.")

syn = pd.read_csv("data/listings.csv")
real = load_real(REAL_FILES_V27)
S, yS = synth_flags(syn, all_flags), syn.label.values
R, yR = real_flags(real, all_flags), real.label.values
texts = real.text.astype(str).values

meaning = MeaningModel(MEANING_C).fit(texts, yR)                         # used by the app
R["meaning_score"] = oof_scores(texts, yR, real.group.values, MEANING_C)  # no leakage in training
S["meaning_score"] = meaning.score(syn.text.astype(str).values)
sem.save_cache()
print(f"Training data: {len(R)} real rows (weight x{REAL_WEIGHT}) + {len(S)} synthetic rows")


def fit(R_, yR_):
    return build_v27().fit(pd.concat([S, R_], ignore_index=True), np.r_[yS, yR_],
                           np.r_[np.ones(len(S)), np.full(len(R_), REAL_WEIGHT)])


model = fit(R, yR)
rng = np.random.RandomState(0)
members = []
for _ in range(N_ENSEMBLE):
    idx = rng.randint(0, len(R), len(R))
    members.append(fit(R.iloc[idx].reset_index(drop=True), yR[idx]))

w = model.weights()
print(f"Learned weight of the meaning score: {w['meaning_score']:.2f} (started at 1.5)")
print("Top signals:", ", ".join(w[w > 0].head(10).index))

Path("model").mkdir(exist_ok=True)
joblib.dump({"model": model, "ensemble": Ensemble(members), "threshold": THRESHOLD, "version": "2.7",
             "meaning": meaning, "meaning_C": MEANING_C,
             "semantic_threshold": sem.SIM_THRESHOLD, "semantic_used": True},
            "model/scam_model.joblib")
print("Saved model/scam_model.joblib (v2.7)")
