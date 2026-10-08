"""
step2_train_model.py - Layer 2: train the scam decision model (v2.4).

v2.4 (after the independent held-out test exposed overfitting):
  * The model now uses the red-flag / good-sign features only (no individual words).
  * It learns mainly from REAL examples - the 120 published-case rows in
    data/real_test_set.csv and data/heldout_test_set.csv, which are now TRAINING
    data - with the synthetic listings as low-weight background.
  * Red flags may only raise the risk, good signs may only lower it.
  Evidence for these choices: python select_hybrid.py (grouped cross-validation).
  Because those 120 rows are now used for training, a NEW blind test set is
  needed to judge v2.4 (see PREREGISTRATION.md).

Outputs:
  model/scam_model.joblib           - the trained model (used by the app)
  outputs/confusion_matrix.png      - synthetic test split (for continuity with v2.3)
  outputs/top_features.png          - "what the model learned" slide (flag weights)
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.model_selection import train_test_split

from channels import channel_info
from features import ALL_DESCRIPTIONS, FLAG_COLUMNS, HARD_STOPS, all_flags
from hybrid_model import REAL_WEIGHT, HybridModel

THRESHOLD = 0.4  # confirmed by grouped cross-validation (0.3 / 0.4 / 0.5 / 0.6 compared)


def tri(v):
    """dataset stores 1 / 0 / -1 (couldn't check) -> True / False / None"""
    return None if v == -1 else bool(v)


def row_to_event(r) -> dict:
    return dict(found=tri(r.event_found), details_match=tri(r.event_details_match),
                high_demand=bool(r.event_high_demand), before_on_sale=bool(r.event_before_on_sale),
                app_only=bool(r.event_app_only))


def row_to_channel(r) -> dict:
    return dict(category=None if r.ch_category == "none" else r.ch_category,
                lookalike=bool(r.ch_lookalike), new_domain=bool(r.ch_new_domain),
                insecure_or_short=bool(r.ch_insecure_or_short))


def _num(v, default):
    return default if pd.isna(v) or v == "" else float(v)


def real_row_flags(r) -> dict:
    """Flags for a real-world row (unknown seller details = neutral; event check off)."""
    p, f = _num(r.price, 0), _num(r.face_value, 0)
    ch = channel_info(r.platform, r.link if isinstance(r.link, str) else "", check_age=False)
    return all_flags(r.text, _num(r.account_age_days, 365), _num(r.followers, 200),
                     p / f if p and f else 1.0, int(r.has_seat_details), int(r.sudden_seller), None, ch)


# ---------------------------------------------------------------- synthetic background data
syn = pd.read_csv("data/listings.csv")
syn_flags = pd.DataFrame([all_flags(r.text, r.account_age_days, r.followers, r.price_ratio,
                                    r.has_seat_details, r.sudden_seller, row_to_event(r), row_to_channel(r))
                          for r in syn.itertuples()])
S = pd.concat([syn[["text"]], syn_flags], axis=1)
yS = syn["label"].values

# ---------------------------------------------------------------- real examples (main teacher)
real = pd.concat([pd.read_csv(f) for f in ["data/real_test_set.csv", "data/heldout_test_set.csv"]
                  if Path(f).exists()], ignore_index=True)
R = pd.concat([real[["text"]], pd.DataFrame([real_row_flags(r) for r in real.itertuples()])], axis=1)
yR = real["label"].values
print(f"Training data: {len(R)} real-world rows (weight x{REAL_WEIGHT}) + {len(S)} synthetic rows (weight x1)")

# ---------------------------------------------------------------- synthetic test split (continuity)
S_tr, S_te, y_tr, y_te = train_test_split(S, yS, test_size=0.25, stratify=yS, random_state=42)
check = HybridModel().fit(pd.concat([S_tr, R], ignore_index=True), np.r_[y_tr, yR],
                          np.r_[np.ones(len(S_tr)), np.full(len(R), REAL_WEIGHT)])
pred = ((check.predict_proba(S_te)[:, 1] >= THRESHOLD) | (S_te[list(HARD_STOPS)].max(axis=1).values == 1)).astype(int)
print(f"Synthetic test split: recall {recall_score(y_te, pred):.0%}, precision {precision_score(y_te, pred):.0%}")
print("Real-world performance: see `python select_hybrid.py` (grouped cross-validation) and the NEW blind test.\n")

Path("outputs").mkdir(exist_ok=True)
disp = ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Genuine", "Scam"],
                                               cmap="Blues", colorbar=False)
disp.ax_.set_title("Synthetic test split (v2.4)")
plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=200); plt.close()

# ---------------------------------------------------------------- final model on everything
model = HybridModel().fit(pd.concat([S, R], ignore_index=True), np.r_[yS, yR],
                          np.r_[np.ones(len(S)), np.full(len(R), REAL_WEIGHT)])

w = model.weights()
w = w[w != 0]
top = w.head(15)[::-1]
plt.figure(figsize=(8, 6))
plt.barh([ALL_DESCRIPTIONS[k][:55] for k in top.index], top.values, color="#2a78d6")
plt.title("What the model learned: weight of each red flag", loc="left")
plt.xlabel("Weight (higher = raises the risk more)")
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("outputs/top_features.png", dpi=200); plt.close()
print("Top signals:", ", ".join(w.head(10).index))
print("Good signs:", ", ".join(w[w < 0].index) or "none")

Path("model").mkdir(exist_ok=True)
joblib.dump({"model": model, "threshold": THRESHOLD, "version": "2.4"}, "model/scam_model.joblib")
print("Saved model/scam_model.joblib and charts in outputs/")
