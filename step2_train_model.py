"""
step2_train_model.py - Layer 2: train and test the scam classifier.

Model: TF-IDF on the listing text + all red-flag features (payment, language,
seller, ticket evidence, event checks) -> logistic regression.
Simple, fast, and explainable (you can show which words/flags push the score up).

Outputs:
  model/scam_model.joblib           - the trained model (used by the app)
  outputs/confusion_matrix.png      - put this on a slide
  outputs/top_features.png          - "what the model learned" slide
  printed precision / recall report - quote these numbers in the video
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features import FLAG_COLUMNS, HARD_STOPS, all_flags

THRESHOLD = 0.4  # below 0.5 on purpose: missing a scam costs more than a false alarm


def tri(v):
    """dataset stores 1 / 0 / -1 (couldn't check) -> True / False / None"""
    return None if v == -1 else bool(v)


def row_to_event(r) -> dict:
    return dict(found=tri(r.event_found), details_match=tri(r.event_details_match),
                high_demand=bool(r.event_high_demand), before_on_sale=bool(r.event_before_on_sale),
                app_only=bool(r.event_app_only))


df = pd.read_csv("data/listings.csv")
flags = df.apply(lambda r: all_flags(r.text, r.account_age_days, r.followers, r.price_ratio,
                                     r.has_seat_details, r.sudden_seller, row_to_event(r)), axis=1)
# (sudden_seller is both a raw column and a flag, so drop raw copies before joining)
df = pd.concat([df.drop(columns=[c for c in FLAG_COLUMNS if c in df.columns]),
                pd.DataFrame(list(flags))], axis=1)

print("How often each flag fires (scam rate when it fires):")
summary = pd.DataFrame({"fires": df[FLAG_COLUMNS].sum(),
                        "scam_rate": [df.loc[df[c] == 1, "label"].mean() for c in FLAG_COLUMNS]})
print(summary.round(2).to_string(), "\n")

X, y = df[["text"] + FLAG_COLUMNS], df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

model = Pipeline([
    ("features", ColumnTransformer([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2), "text"),
        ("flags", "passthrough", FLAG_COLUMNS),
    ])),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)

# ---- Evaluate: model alone ----
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= THRESHOLD).astype(int)
print("MODEL ONLY")
print(classification_report(y_test, pred, target_names=["genuine", "scam"]))
print(f"Scam recall (share of scams caught): {recall_score(y_test, pred):.0%}")
print(f"Scam precision (share of alerts that were real scams): {precision_score(y_test, pred):.0%}\n")

# ---- Evaluate: model + hard-stop rules ----
hard = X_test[list(HARD_STOPS)].max(axis=1).values
pred_hs = ((pred == 1) | (hard == 1)).astype(int)
print("MODEL + HARD-STOP RULES")
print(f"Scam recall: {recall_score(y_test, pred_hs):.0%}   "
      f"precision: {precision_score(y_test, pred_hs):.0%}\n")

Path("outputs").mkdir(exist_ok=True)
disp = ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Genuine", "Scam"],
                                               cmap="Blues", colorbar=False)
disp.ax_.set_title("Ticket scam checker - test set results")
plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=200); plt.close()

# ---- What did the model learn? (top features pushing towards "scam") ----
names = model.named_steps["features"].get_feature_names_out()
coefs = model.named_steps["clf"].coef_[0]
top = (pd.Series(coefs, index=[n.split("__", 1)[1] for n in names])
       .sort_values(ascending=False).head(15)[::-1])
top.plot.barh(figsize=(7, 6), color="#c0392b")
plt.title("Top signals the model links to scams"); plt.xlabel("Model weight")
plt.tight_layout(); plt.savefig("outputs/top_features.png", dpi=200); plt.close()
print("Top 15 signals:", ", ".join(top.index[::-1]))

Path("model").mkdir(exist_ok=True)
joblib.dump({"model": model, "threshold": THRESHOLD}, "model/scam_model.joblib")
print("Saved model/scam_model.joblib and charts in outputs/")
