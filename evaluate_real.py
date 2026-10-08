"""
evaluate_real.py - does the checker work on REAL scam examples, and does the
hybrid design beat each of its parts on their own?

Run AFTER step1 + step2 + build_real_test_set.py:
    python evaluate_real.py            # rules, text model, hybrid (development set)
    python evaluate_real.py --test data/heldout_test_set.csv   # your independent held-out set
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
import argparse
_args = argparse.ArgumentParser()
_args.add_argument("--test", default="data/real_test_set.csv",
                   help="test file, e.g. data/heldout_test_set.csv for the independent held-out set")
TEST_FILE = _args.parse_args().test
test = pd.read_csv(TEST_FILE)
if "source_type" not in test:
    test["source_type"] = "heldout"
if "source" not in test:
    test["source"] = ""
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
preds["Rules only (v2.4)"] = ((hard == 1) | (n_meaningful >= 2)).astype(int).values
scores["Rules only (v2.4)"] = np.minimum(n_meaningful / 4, 1).values

# ---------------------------------------------------------------- 1b. rules only, FROZEN v2.2
# The same decision rule, but with the rulebook as it was BEFORE the development set
# existed - the only rules result here that was not influenced by seeing these examples.
import rules_v22_frozen as v22


def row_flags_v22(r) -> dict:
    price, face = num(r.price, 0), num(r.face_value, 0)
    channel = channel_info(r.platform, r.link if isinstance(r.link, str) else "", check_age=False)
    return v22.all_flags(r.text, num(r.account_age_days, NEUTRAL_AGE), num(r.followers, NEUTRAL_FOLLOWERS),
                         price / face if price and face else 1.0, int(r.has_seat_details),
                         int(r.sudden_seller), None, channel)


flags22 = pd.DataFrame([row_flags_v22(r) for r in test.itertuples()])
m22 = [c for c in v22.FLAG_COLUMNS if c not in WEAK_FLAGS and c not in v22.PROTECTIVE]
hard22 = flags22[list(v22.HARD_STOPS)].max(axis=1)
preds["Rules only (v2.2 frozen)"] = ((hard22 == 1) | (flags22[m22].sum(axis=1) >= 2)).astype(int).values
scores["Rules only (v2.2 frozen)"] = np.minimum(flags22[m22].sum(axis=1) / 4, 1).values

# ---------------------------------------------------------------- 2. text model only
text_model = Pipeline([("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                       ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
text_model.fit(synthetic["text"], synthetic["label"])
scores["Text model only"] = text_model.predict_proba(test["text"])[:, 1]
preds["Text model only"] = (scores["Text model only"] >= THRESHOLD).astype(int)

# ---------------------------------------------------------------- 4. hybrid (the real checker)
bundle = joblib.load("model/scam_model.joblib")
if bundle.get("version") == "2.4" and not TEST_FILE.endswith("blind2_test_set.csv"):
    raise SystemExit(f"Stopped: the v2.4 model was trained on {TEST_FILE}, so testing on it would be invalid "
                     "and would overwrite your earlier results.\nUse: python evaluate_real.py --test "
                     "data/blind2_test_set.csv")
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
    # saved answers, so re-runs don't spend your free quota again
    cache_file = OUT / ("llm_cache.json" if TEST_FILE.endswith("real_test_set.csv")
                        else f"llm_cache_{Path(TEST_FILE).stem}.json")
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    probs, failed = [], []
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
            if answer is None:  # not cached, so a re-run will try this listing again
                failed.append(r.id)
                print(f"  {r.id}: AI call failed ({last.__class__.__name__})")
                probs.append(np.nan)
                continue
            cache[r.id] = answer
            cache_file.write_text(json.dumps(cache, indent=1))
        probs.append(float(cache[r.id]["scam_probability"]))
    if failed:
        print(f"\nAI-only arm INCOMPLETE: {len(failed)} listings failed (often the free-tier limit). "
              "Wait a while and run evaluate_real.py again - finished answers are saved, so only "
              "the missing ones are retried. The AI-only row is left out until all succeed.")
        return None
    return np.array(probs)


ai = llm_only()
if ai is not None:
    scores["AI only (Gemini)"] = ai
    preds["AI only (Gemini)"] = (ai >= 0.5).astype(int)

DEV_SET = TEST_FILE.endswith("real_test_set.csv")
if not DEV_SET:  # held-out run: the same four approaches as the original comparison
    preds.pop("Rules only (v2.2 frozen)"); scores.pop("Rules only (v2.2 frozen)")
ORDER = [k for k in ["Rules only (v2.2 frozen)", "Rules only (v2.4)", "Text model only", "AI only (Gemini)", "Hybrid (our checker)"] if k in preds]


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
_tag = "real_eval" if DEV_SET else Path(TEST_FILE).stem.replace("_test_set", "") + "_eval"
detail.to_csv(OUT / f"{_tag}_predictions.csv", index=False)
summary.to_csv(OUT / f"{_tag}_summary.csv")

# ---------------------------------------------------------------- chart
fig, ax = plt.subplots(figsize=(10, 4.4))
colors = {"Scams caught (recall)": "#2a78d6", "Alerts correct (precision)": "#eb6834", "F1": "#1baf7a"}
vals = {"Scams caught (recall)": summary["recall"], "Alerts correct (precision)": summary["precision"],
        "F1": summary["f1"]}
w, xs = 0.26, np.arange(len(ORDER))
for i, (name, v) in enumerate(vals.items()):
    bars = ax.bar(xs + (i - 1) * w, v.astype(float), w - 0.02, color=colors[name], label=name)
    for b, val in zip(bars, v.astype(float)):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.015, f"{val:.0%}", ha="center", va="bottom",
                fontsize=8, color="#333333")
ax.set_xticks(xs, [o.replace(" (", "\n(") for o in ORDER], fontsize=9)
ax.set_ylim(0, 1.2)
ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8)
ax.set_title(("Development set" if DEV_SET else "Independent held-out set") + f" ({int(y.sum())} scams, {int((1 - y).sum())} genuine)", fontsize=11, loc="left")
plt.tight_layout()
plt.savefig(OUT / f"{_tag}_chart.png", dpi=200)
plt.close()


# ---------------------------------------------------------------- report
def pct(v):
    return f"{float(v):.0%}"


h = "Hybrid (our checker)"
sec_pred = ((scores[h] >= 0.30) | (hard == 1)).astype(int).values
sec = metrics(y, sec_pred)
misses = detail[(detail.label == 1) & (detail[h] == 0)]
false_alarms = detail[(detail.label == 0) & (detail[h] == 1)]
DEV_BLOCK = [
    "## Version history on the development set",
    "- v2.2 (before fixes): Rules 30/40 (F1 79%), Text model 38/40 (F1 69%), AI only 39/40 (F1 93%), "
    "Hybrid 19/40 (F1 62%, 2 false alarms). Cause: synthetic scams always carried several obvious flags, "
    "while real ones are short and vague; several real tactics had no rule.",
    "- v2.3: widened rules (pm me, up front, third-party accounts, gift-card wording, presale-code sales, "
    "like-and-comment bait, screenshots as proof, vague posts) and made the training data more realistic "
    "(short vague scams, short genuine posts, missing prices). Results above.",
    f"- Caution: v2.3 was built after studying this set's mistakes, so results on {TEST_FILE} "
    "are optimistic unless it is a fresh held-out set. The fair test is a new set collected independently "
    "(see data/heldout_template.csv).",
    "",
]
lines = [
    "# Real-world evaluation",
    "",
    f"Test file: `{TEST_FILE}`",
    "",
    f"Test set: {len(test)} listings ({int(y.sum())} scams from published sources: "
    f"{int((test.source_type == 'verbatim').sum())} verbatim, {int((test.source_type == 'reconstructed').sum())} "
    f"reconstructed from documented UK cases; {int((1 - y).sum())} constructed genuine listings). "
    "None were used in training.",
    "",
    "'Rules only (v2.2 frozen)' uses the rulebook written BEFORE this test set existed; 'Rules only (v2.4)' uses "
    "the current v2.4 rulebook. On the development set, only the frozen version is an untuned result.",
    "",
    "| Approach | Scams caught | Recall | Precision | F1 | False alarms |",
    "|---|---|---|---|---|---|",
    *[f"| {k} | {r.scams_caught} | {pct(r.recall)} | {pct(r.precision)} | {pct(r.f1)} | {r.false_alarms} |"
      for k, r in summary.iterrows()],
    "",
    f"For comparison, the hybrid on its own synthetic test set: recall {pct(syn['recall'])}, "
    f"precision {pct(syn['precision'])}, F1 {pct(syn['f1'])}.",
    "",
    *(["Recall by source type:", "",
       "| Approach | Verbatim quotes | Reconstructed cases |", "|---|---|---|",
       *[f"| {k} | {pct(r.verbatim)} | {pct(r.reconstructed)} |" for k, r in by_type.iterrows()]]
      if (test.source_type == "verbatim").any() else []),
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
    *(DEV_BLOCK if DEV_SET else [
        "## Held-out protocol",
        "- The checker, model, rules, thresholds and Gemini prompt were frozen and recorded in the "
        "PREREGISTRATION file "
        "BEFORE this set existed.",
        "- The set was built by an independent agent that had not seen the code, rules, training data or "
        "development set. It was run once; nothing was changed afterwards.",
        f"- Pre-declared secondary result, hybrid at threshold 0.30: scams caught {sec['scams_caught']}, "
        f"false alarms {sec['false_alarms']}, F1 {pct(sec['f1'])}.",
        ""]),
    "## Method notes",
    "- Thresholds and the rules-only decision rule were fixed before running the test.",
    "- Event check and website-age lookup were switched off for all approaches (see docstring).",
    "- Unknown seller details were treated as neutral (365-day account, 200 followers).",
    "- Limitations: small test set; genuine listings are constructed, not collected; reconstructed scams were "
    "written by the developer, which risks unconscious bias towards the rules. Next step: an independent, "
    "real-world labelled set via a bank or platform partner.",
]
(OUT / f"{_tag}_report.md").write_text("\n".join(lines), encoding="utf-8")

print(summary[["scams_caught", "recall", "precision", "f1", "false_alarms"]].to_string())
print(f"\nHybrid on synthetic test: recall {pct(syn['recall'])}, precision {pct(syn['precision'])}")
print(f"\nSaved outputs/{_tag}_report.md, {_tag}_chart.png, {_tag}_summary.csv, {_tag}_predictions.csv")
