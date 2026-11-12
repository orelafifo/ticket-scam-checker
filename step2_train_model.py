"""
step2_train_model.py - Layer 2: train the scam decision model (v2.6).

v2.6 changes to the INPUTS (the learning rule is the same as v2.5):
  * text is normalised first (normalise.py) and the rulebook knows more wordings
    (DM / inbox / hmu, Revolut / Monzo / TWINT / PayPal F&F...)
  * soft red flags can also be found by MiniLM, the AI meaning-matcher (semantic_flags.py)
  * 3 new flags from "How did you come across this seller?" (group chat with strangers,
    seller messaged first, friend's account). No training row records this yet, so their
    weights stay at the expert tier (the v2.5 informative prior does exactly that).
v2.6 also trains an ENSEMBLE: the same model refitted on 30 bootstrap resamples of the real
examples. The spread of the 30 scores tells the app when the model is unsure ("Borderline").

v2.5: each red flag's weight starts from an expert evidence tier and training adjusts it
(C = 0.3). v2.4: flag features only, real examples weighted x50, sign constraints.

Outputs:
  model/scam_model.joblib     - model + ensemble + threshold (used by the app)
  outputs/top_features.png    - "what the model learned" (flag weights)
  outputs/confusion_matrix.png - synthetic test split (continuity only - not a real-world result)
"""
import textwrap
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.model_selection import train_test_split

import semantic_flags as sem
from data_utils import load_real, real_flags, synth_flags
from features import ALL_DESCRIPTIONS, HARD_STOPS, all_flags
from hybrid_model import REAL_WEIGHT, Ensemble, build_v26

THRESHOLD = 0.4      # confirmed by grouped cross-validation in v2.4
N_ENSEMBLE = 30

print("MiniLM meaning-matcher:", "ON" if sem.available() else "OFF (rules only - check internet / install)")
syn = pd.read_csv("data/listings.csv")
S, yS = synth_flags(syn, all_flags), syn.label.values
real = load_real()
R, yR = real_flags(real, all_flags), real.label.values
sem.save_cache()
print(f"Training data: {len(R)} real-world rows (weight x{REAL_WEIGHT}) + {len(S)} synthetic rows (weight x1)")


def fit(S_, yS_, R_, yR_):
    return build_v26().fit(pd.concat([S_, R_], ignore_index=True), np.r_[yS_, yR_],
                           np.r_[np.ones(len(S_)), np.full(len(R_), REAL_WEIGHT)])


# ---------------------------------------------------------------- synthetic test split (continuity only)
S_tr, S_te, y_tr, y_te = train_test_split(S, yS, test_size=0.25, stratify=yS, random_state=42)
check = fit(S_tr, y_tr, R, yR)
pred = ((check.predict_proba(S_te)[:, 1] >= THRESHOLD) | (S_te[list(HARD_STOPS)].max(axis=1).values == 1)).astype(int)
print(f"Synthetic test split: recall {recall_score(y_te, pred):.0%}, precision {precision_score(y_te, pred):.0%} "
      "(synthetic - not a real-world result; see select_v26.py)")
Path("outputs").mkdir(exist_ok=True)
disp = ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Genuine", "Scam"],
                                               cmap="Blues", colorbar=False)
disp.ax_.set_title("Synthetic test split (v2.6)")
plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=200); plt.close()

# ---------------------------------------------------------------- final model + bootstrap ensemble
model = fit(S, yS, R, yR)
rng = np.random.RandomState(0)
members = []
for _ in range(N_ENSEMBLE):
    idx = rng.randint(0, len(R), len(R))           # resample the real examples with replacement
    members.append(fit(S, yS, R.iloc[idx].reset_index(drop=True), yR[idx]))
ensemble = Ensemble(members)

# ---------------------------------------------------------------- "what the model learned" chart
w = model.weights()
w = w[w != 0]
top = w.head(15)[::-1]
plt.figure(figsize=(10, 7))
labels = ["\n".join(textwrap.wrap(ALL_DESCRIPTIONS[k], 48)) for k in top.index]
plt.barh(labels, top.values, color="#2a78d6")
plt.title("What the model learned: weight of each red flag (v2.6)", loc="left")
plt.xlabel("Weight (higher = raises the risk more)")
plt.yticks(fontsize=8)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("outputs/top_features.png", dpi=200); plt.close()
print("Top signals:", ", ".join(w.head(10).index))
print("Good signs:", ", ".join(w[w < 0].index) or "none")

Path("model").mkdir(exist_ok=True)
joblib.dump({"model": model, "ensemble": ensemble, "threshold": THRESHOLD, "version": "2.6",
             "semantic_threshold": sem.SIM_THRESHOLD, "semantic_used": sem.available()},
            "model/scam_model.joblib")
print("Saved model/scam_model.joblib and charts in outputs/")
