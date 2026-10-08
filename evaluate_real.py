"""
evaluate_real.py - does the checker work on REAL scam examples, and does the
hybrid design beat each of its parts on their own?

Run AFTER step1 + step2 + build_real_test_set.py:
    python evaluate_real.py            # rules, text model, hybrid
    (with GEMINI_API_KEY in .env, the "AI only" arm runs too)

Four approaches, all judged on data/real_test_set.csv (never used in training):
  1. Rules only   - hand-written red-flag rules, no machine learning.
                    Scam if any hard-stop rule fires OR 2+ meaningful red flags.
  2. Text model   - machine learning on the listing wording only (TF-IDF +
                    logistic regression, trained on the synthetic data, no rules).
  3. AI only      - an LLM (Gemini) asked directly "is this a scam?", zero-shot.
  4. Hybrid       - the actual checker: rule flags + wording in one model,
                    plus hard-stop rules. (The LLM only explains, it never decides.)

All decision rules and thresholds were fixed BEFORE looking at the results.
The event check is switched off for every approach, because these are past or
example events that can't be looked up fairly. Website age (RDAP) is also off,
because several test links are invented examples.

Outputs (in outputs/):
  real_eval_summary.csv      - one row per approach
  real_eval_predictions.csv  - every listing, every approach, which flags fired
  real_eval_chart.png        - chart for the report
  real_eval_report.md        - tables + error analysis, ready to quote
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from channels import channel_info
from features import FLAG_COLUMNS, HARD_STOPS, PROTECTIVE, all_flags

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
THRESHOLD = 0.4                                   # same threshold as the live app
WEAK_FLAGS = {"no_seat_details", "high_demand"}   # too common to count on their own (decided up front)
NEUTRAL_AGE, NEUTRAL_FOLLOWERS = 365, 200         # used when a source doesn't say


# ---------------------------------------------------------------- load data
test = pd.read_csv("data/real_test_set.csv")
synthetic = pd.read_csv("data/listings.csv")


def num(v, default):
    return default if pd.isna(v) or v == "" else float(v)


def row_flags(r) -> dict:
    price, face = num(r.price, 0), num(r.face_value, 0)
    channel = channel_info(r.platform, r.link if isinstance(r.link, str) else "", check_age=False)
    return all_flags(r.text, num(r.account_age_days, NEUTRAL_AGE), num(r.followers, NEUTRAL_FOLLOWERS),
                     price / face if price and face else 1.0, int(r.has_seat_details),
                     int(r.sudden_seller), None, channel)


flags = pd.DataFrame([row_flags(r) for r in test.itertuples()])
y = test["label"].values
preds, scores = {}, {}

# ---------------------------------------------------------------- 1. rules only
meaningful = [c for c in FLAG_COLUMNS if c not in WEAK_FLAGS and c not in PROTECTIVE]
n_meaningful = flags[meaningful].sum(axis=1)
hard = flags[list(HARD_STOPS)].max(axis=1)
preds["Rules only"] = ((hard == 1) | (n_meaningful >= 2)).astype(int).values
scores["Rules only"] = np.minimum(n_meaningful / 4, 1).values

# ---------------------------------------------------------------- 2. text model only
text_model = Pipeline([("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                       ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
text_model.fit(synthetic["text"], synthetic["label"])
scores["Text model only"] = text_model.predict_proba(test["text"])[:, 1]
preds["Text model only"] = (scores["Text model only"] >= THRESHOLD).astype(int)

# ---------------------------------------------------------------- 4. hybrid (the real checker)
bundle = joblib.load("model/scam_model.joblib")
hybrid = bundle["model"]
X = pd.concat([test[["text"]], flags], axis=1)[["text"] + FLAG_COLUMNS]
scores["Hybrid (our checker)"] = hybrid.predict_proba(X)[:, 1]
preds["Hybrid (our checker)"] = ((scores["Hybrid (our checker)"] >= THRESHOLD) | (hard == 1)).astype(int).values

# ---------------------------------------------------------------- 3. AI only (optional)
LLM_PROMPT = """You are checking a resale ticket listing for fraud.
Where it is being sold: {platform}{link}
Listing / messages:
\"\"\"{text}\"\"\"
Reply with JSON only: {{"scam_probability": <number 0 to 1>, "reason": "<max 15 words>"}}"""


def llm_only() -> np.ndarray | None:
    if not os.getenv("GEMINI_API_KEY"):
        print("AI-only arm skipped (no GEMINI_API_KEY). Add it to .env and re-run to include it.")
        return None
    from explainer import GEMINI_MODELS
    from google import genai
    client = genai.Client()
    cache_file = OUT / "llm_cache.json"   # so re-runs don't spend your free quota again
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    probs = []
    for r in test.itertuples():
        if r.id not in cache:
            link = f" (link: {r.link})" if isinstance(r.link, str) and r.link else ""
            prompt = LLM_PROMPT.format(platform=r.platform, link=link, text=r.text)
            answer = None
            for model in GEMINI_MODELS:
                try:
                    resp = client.models.generate_content(
                        model=model, contents=prompt,
                        config=genai.types.GenerateContentConfig(max_output_tokens=2048,
                                                                 response_mime_type="application/json"))
                    answer = json.loads(resp.text)
                    answer["model"] = model
                    break
                except Exception as e:
                    last = e
                    continue
            if answer is None:
                print(f"  {r.id}: AI call failed ({last.__class__.__name__}), counted as 0.5")
                answer = {"scam_probability": 0.5, "reason": "call failed", "model": None}
            cache[r.id] = answer
            cache_file.write_text(json.dumps(cache, indent=1))
        probs.append(float(cache[r.id]["scam_probability"]))
    return np.array(probs)


ai = llm_only()
if ai is not None:
    scores["AI only (Gemini)"] = ai
    preds["AI only (Gemini)"] = (ai >= 0.5).astype(int)

ORDER = [k for k in ["Rules only", "Text model only", "AI only (Gemini)", "Hybrid (our checker)"] if k in preds]


# ---------------------------------------------------------------- metrics
def metrics(y_true, y_pred) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum()); fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(scams_caught=f"{tp}/{tp + fn}", recall=rec, precision=prec, f1=f1,
                false_alarms=f"{fp}/{fp + tn}", false_alarm_rate=fp / (fp + tn), accuracy=(tp + tn) / len(y_true))


summary = pd.DataFrame({k: metrics(y, preds[k]) for k in ORDER}).T
by_type = pd.DataFrame({k: {t: preds[k][(test.source_type == t).values & (y == 1)].mean()
                            for t in ["verbatim", "reconstructed"]} for k in ORDER}).T

# synthetic-vs-real: how much does the hybrid drop outside its own data?
# (rebuilds the same features and the same train/test split as step2_train_model.py)
def _tri(v):
    return None if v == -1 else bool(v)


syn_rows = [all_flags(r.text, r.account_age_days, r.followers, r.price_ratio, r.has_seat_details,
                      r.sudden_seller,
                      dict(found=_tri(r.event_found), details_match=_tri(r.event_details_match),
                           high_demand=bool(r.event_high_demand), before_on_sale=bool(r.event_before_on_sale),
                           app_only=bool(r.event_app_only)),
                      dict(category=None if r.ch_category == "none" else r.ch_category,
                           lookalike=bool(r.ch_lookalike), new_domain=bool(r.ch_new_domain),
                           insecure_or_short=bool(r.ch_insecure_or_short)))
            for r in synthetic.itertuples()]
syn_X = pd.concat([synthetic[["text"]], pd.DataFrame(syn_rows)], axis=1)[["text"] + FLAG_COLUMNS]
_, X_syn_test, _, y_syn_test = train_test_split(syn_X, synthetic["label"], test_size=0.25,
                                                stratify=synthetic["label"], random_state=42)
syn_hard = X_syn_test[list(HARD_STOPS)].max(axis=1).values
syn_pred = ((hybrid.predict_proba(X_syn_test)[:, 1] >= THRESHOLD) | (syn_hard == 1)).astype(int)
syn = metrics(y_syn_test.values, syn_pred)

# ---------------------------------------------------------------- per-listing table
detail = test[["id", "label", "source_type", "source", "platform", "text"]].copy()
for k in ORDER:
    detail[k] = preds[k]
detail["hybrid_score"] = scores["Hybrid (our checker)"].round(2)
detail["flags_fired"] = flags.apply(lambda f: ", ".join(c for c in FLAG_COLUMNS if f[c]), axis=1)
detail.to_csv(OUT / "real_eval_predictions.csv", index=False)
summary.to_csv(OUT / "real_eval_summary.csv")

# ---------------------------------------------------------------- chart
fig, ax = plt.subplots(figsize=(8, 4.2))
colors = {"Scams caught (recall)": "#2a78d6", "Alerts correct (precision)": "#eb6834", "F1": "#1baf7a"}
vals = {"Scams caught (recall)": summary["recall"], "Alerts correct (precision)": summary["precision"],
        "F1": summary["f1"]}
w, xs = 0.26, np.arange(len(ORDER))
for i, (name, v) in enumerate(vals.items()):
    bars = ax.bar(xs + (i - 1) * w, v.astype(float), w - 0.02, color=colors[name], label=name)
    for b, val in zip(bars, v.astype(float)):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.015, f"{val:.0%}", ha="center", va="bottom",
                fontsize=8, color="#333333")
ax.set_xticks(xs, ORDER)
ax.set_ylim(0, 1.12)
ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8)
ax.set_title(f"Real-world test set ({int(y.sum())} scams, {int((1 - y).sum())} genuine)", fontsize=11, loc="left")
plt.tight_layout()
plt.savefig(OUT / "real_eval_chart.png", dpi=200)
plt.close()


# ---------------------------------------------------------------- report
def pct(v):
    return f"{float(v):.0%}"


h = "Hybrid (our checker)"
misses = detail[(detail.label == 1) & (detail[h] == 0)]
false_alarms = detail[(detail.label == 0) & (detail[h] == 1)]
lines = [
    "# Real-world evaluation",
    "",
    f"Test set: {len(test)} listings ({int(y.sum())} scams from published sources: "
    f"{int((test.source_type == 'verbatim').sum())} verbatim, {int((test.source_type == 'reconstructed').sum())} "
    f"reconstructed from documented UK cases; {int((1 - y).sum())} constructed genuine listings). "
    "None were used in training.",
    "",
    "| Approach | Scams caught | Recall | Precision | F1 | False alarms |",
    "|---|---|---|---|---|---|",
    *[f"| {k} | {r.scams_caught} | {pct(r.recall)} | {pct(r.precision)} | {pct(r.f1)} | {r.false_alarms} |"
      for k, r in summary.iterrows()],
    "",
    f"For comparison, the hybrid on its own synthetic test set: recall {pct(syn['recall'])}, "
    f"precision {pct(syn['precision'])}, F1 {pct(syn['f1'])}.",
    "",
    "Recall by source type:",
    "",
    "| Approach | Verbatim quotes | Reconstructed cases |",
    "|---|---|---|",
    *[f"| {k} | {pct(r.verbatim)} | {pct(r.reconstructed)} |" for k, r in by_type.iterrows()],
    "",
    f"## Scams the hybrid missed ({len(misses)})",
    "",
    *[f"- **{r.id}** ({r.source}; score {r.hybrid_score}): \"{r.text[:140]}\" - flags: {r.flags_fired or 'none'}"
      for r in misses.itertuples()],
    "",
    f"## Genuine listings the hybrid wrongly flagged ({len(false_alarms)})",
    "",
    *[f"- **{r.id}** ({r.platform}; score {r.hybrid_score}): \"{r.text[:140]}\" - flags: {r.flags_fired or 'none'}"
      for r in false_alarms.itertuples()],
    "",
    "## What this shows (interpretation, written after the v1 run)",
    "- The synthetic results do not carry over to real examples: the hybrid drops sharply on real scams. "
    "This is the expected effect of training only on template data (a 'distribution shift').",
    "- Real scam posts are often short and vague (\"Selling 4x tickets pm me\"), with no price or payment detail. "
    "The synthetic scams almost always carried several obvious flags, so the model learned that one or two "
    "weak signals mean 'genuine'.",
    "- Some real tactics are not covered by any rule yet: 'pm me', paying 'upfront', paying into a friend's "
    "account, gift-card wording, registration fees for codes, 'like and comment first', vague 'reduced price' posts.",
    "- The text-only model catches almost everything but raises a false alarm on most genuine posts, so it is "
    "not usable alone; rules alone are the most balanced today.",
    "- Next iteration: widen the rules, make the synthetic training data more realistic (short vague scams, "
    "messy genuine sellers), then re-test on a NEW held-out set collected independently. This 80-row set has "
    "now been used for error analysis, so it becomes the development set.",
    "",
    "## Method notes",
    "- Thresholds and the rules-only decision rule were fixed before running the test.",
    "- Event check and website-age lookup were switched off for all approaches (see docstring).",
    "- Unknown seller details were treated as neutral (365-day account, 200 followers).",
    "- Limitations: small test set; genuine listings are constructed, not collected; reconstructed scams were "
    "written by the developer, which risks unconscious bias towards the rules. Next step: an independent, "
    "real-world labelled set via a bank or platform partner.",
]
(OUT / "real_eval_report.md").write_text("\n".join(lines), encoding="utf-8")

print(summary[["scams_caught", "recall", "precision", "f1", "false_alarms"]].to_string())
print(f"\nHybrid on synthetic test: recall {pct(syn['recall'])}, precision {pct(syn['precision'])}")
print("\nSaved outputs/real_eval_report.md, real_eval_chart.png, real_eval_summary.csv, real_eval_predictions.csv")
