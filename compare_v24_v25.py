"""
compare_v24_v25.py - blind test #3: v2.4 vs v2.5 on the SAME new, unseen examples.

Pre-registered in PREREGISTRATION_v25.md BEFORE data/blind3_test_set.csv existed.
Run it ONCE. Do not change rules, tiers, C, threshold or prompt after seeing the result.

Approaches compared (same 40 examples):
  1. Rules only (v2.5 rulebook)
  2. Text model only (TF-IDF, synthetic listings, threshold 0.4)
  3. AI only (Gemini, threshold 0.5) - only if GEMINI_API_KEY is in .env;
     answers saved in outputs/llm_cache_blind3_test_set.json
  4. Hybrid v2.4 - rebuilt exactly (hybrid_model.build_v24)
  5. Hybrid v2.5 - the saved model (model/scam_model.joblib)
Event check and website-age lookup are off for all approaches (as in earlier tests).

Run:  python compare_v24_v25.py
Outputs (outputs/):
  blind3_report.md        - table ready to paste + every example's result
  blind3_chart.png        - all approaches
  blind3_hybrid_v24_vs_v25.png - the two hybrids side by side
  blind3_results.csv, blind3_predictions.csv
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
from sklearn.pipeline import Pipeline

import features as rules
from channels import channel_info
from hybrid_model import REAL_WEIGHT, build_v24

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
TEST = "data/blind3_test_set.csv"
THRESHOLD = 0.4
WEAK = {"no_seat_details", "high_demand"}


def tri(v):
    return None if v == -1 else bool(v)


def num(v, d):
    return d if pd.isna(v) or v == "" else float(v)


def synth_X(df):
    return pd.DataFrame([rules.all_flags(
        r.text, r.account_age_days, r.followers, r.price_ratio, r.has_seat_details, r.sudden_seller,
        dict(found=tri(r.event_found), details_match=tri(r.event_details_match),
             high_demand=bool(r.event_high_demand), before_on_sale=bool(r.event_before_on_sale),
             app_only=bool(r.event_app_only)),
        dict(category=None if r.ch_category == "none" else r.ch_category, lookalike=bool(r.ch_lookalike),
             new_domain=bool(r.ch_new_domain), insecure_or_short=bool(r.ch_insecure_or_short)))
        for r in df.itertuples()])


def real_X(df):
    rows = []
    for r in df.itertuples():
        p, f = num(r.price, 0), num(r.face_value, 0)
        ch = channel_info(r.platform, r.link if isinstance(r.link, str) else "", check_age=False)
        rows.append(rules.all_flags(r.text, num(r.account_age_days, 365), num(r.followers, 200),
                                    p / f if p and f else 1.0, int(num(r.has_seat_details, 0)),
                                    int(num(r.sudden_seller, 0)), None, ch))
    return pd.DataFrame(rows)


def hard(X):
    return (X[list(rules.HARD_STOPS)].max(axis=1) == 1).values


def rules_only(X):
    meaningful = [c for c in rules.FLAG_COLUMNS if c not in WEAK and c not in rules.PROTECTIVE]
    return (hard(X) | (X[meaningful].sum(axis=1) >= 2).values).astype(int)


def text_only(texts):
    syn = pd.read_csv("data/listings.csv")
    m = Pipeline([("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                  ("c", LogisticRegression(max_iter=1000, class_weight="balanced"))]).fit(syn.text, syn.label)
    return (m.predict_proba(texts)[:, 1] >= THRESHOLD).astype(int)


def hybrid_v24(X):
    syn = pd.read_csv("data/listings.csv")
    real = pd.concat([pd.read_csv("data/real_test_set.csv"), pd.read_csv("data/heldout_test_set.csv")],
                     ignore_index=True)
    S, R = synth_X(syn), real_X(real)
    m = build_v24().fit(pd.concat([S, R], ignore_index=True), np.r_[syn.label.values, real.label.values],
                        np.r_[np.ones(len(S)), np.full(len(R), REAL_WEIGHT)])
    p = m.predict_proba(X)[:, 1]
    return ((p >= THRESHOLD) | hard(X)).astype(int), p


def hybrid_v25(X):
    saved = joblib.load("model/scam_model.joblib")
    if saved.get("version") != "2.5":
        raise SystemExit("model/scam_model.joblib is not v2.5 - run python step2_train_model.py first.")
    p = saved["model"].predict_proba(X)[:, 1]
    return ((p >= saved["threshold"]) | hard(X)).astype(int), p


LLM_PROMPT = """You are checking a resale ticket listing for fraud.
Where it is being sold: {platform}{link}
Listing / messages:
\"\"\"{text}\"\"\"
Reply with JSON only: {{"scam_probability": <number 0 to 1>, "reason": "<max 15 words>"}}"""


def gemini_only(test):
    if not os.getenv("GEMINI_API_KEY"):
        print("AI only (Gemini) skipped: add GEMINI_API_KEY to .env and run again to include it.")
        return None
    from google import genai
    from explainer import GEMINI_MODELS
    client = genai.Client()
    cache_file = OUT / f"llm_cache_{Path(TEST).stem}.json"
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
                    answer = json.loads(resp.text); answer["model"] = model
                    break
                except Exception:
                    continue
            if answer is None:
                failed.append(r.id); probs.append(np.nan)
                print(f"  {r.id}: Gemini call failed")
                continue
            cache[r.id] = answer
            cache_file.write_text(json.dumps(cache, indent=1))
        probs.append(float(cache[r.id]["scam_probability"]))
    if failed:
        print(f"Gemini incomplete ({len(failed)} failed). Run again later - saved answers are reused.")
        return None
    return (np.array(probs) >= 0.5).astype(int)


def metrics(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(y.sum()) - tp; neg = int((1 - y).sum())
    return {"Scams caught": f"{tp}/{int(y.sum())}", "Alerts correct": tp / max(tp + fp, 1),
            "F1": 2 * tp / max(2 * tp + fp + fn, 1), "False alarms": f"{fp}/{neg}", "recall": tp / y.sum()}


# ---------------------------------------------------------------- run once
test = pd.read_csv(TEST)
y = test.label.values
X = real_X(test)
p24, s24 = hybrid_v24(X)
p25, s25 = hybrid_v25(X)
preds = {"Rules only": rules_only(X), "Text model only": text_only(test.text)}
gem = gemini_only(test)
if gem is not None:
    preds["AI only (Gemini)"] = gem
preds["Hybrid v2.4"] = p24
preds["Hybrid v2.5"] = p25
res = pd.DataFrame({k: metrics(y, p) for k, p in preds.items()}).T


def chart(df, title, file):
    cols = {"Scams caught (recall)": ("recall", "#2a78d6"), "Alerts correct (precision)": ("Alerts correct", "#eb6834"),
            "F1": ("F1", "#1baf7a")}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    xs, w = np.arange(len(df)), 0.26
    for i, (label, (col, colr)) in enumerate(cols.items()):
        vals = df[col].astype(float)
        bars = ax.bar(xs + (i - 1) * w, vals, w - 0.02, color=colr, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=8, color="#333333")
    ax.set_xticks(xs, df.index)
    ax.set_ylim(0, 1.2); ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=10, loc="left")
    plt.tight_layout(); plt.savefig(OUT / file, dpi=200); plt.close()


n_s, n_g = int(y.sum()), int((1 - y).sum())
chart(res, f"Blind test #3 ({n_s} scams, {n_g} genuine) - new examples, run once", "blind3_chart.png")
chart(res.loc[["Hybrid v2.4", "Hybrid v2.5"]], "Hybrid v2.4 vs v2.5 - same unseen examples (blind test #3)",
      "blind3_hybrid_v24_vs_v25.png")
res.drop(columns="recall").to_csv(OUT / "blind3_results.csv")

per = test[["id", "label", "source_type", "platform", "source"]].copy()
per["v24_score"], per["v25_score"] = s24.round(2), s25.round(2)
per["hard_stop"] = hard(X)
for k, p in preds.items():
    per[k] = p
per.to_csv(OUT / "blind3_predictions.csv", index=False)
diff = per[per["Hybrid v2.4"] != per["Hybrid v2.5"]]

rows = ["| Approach | Scams caught | Alerts correct | F1 | False alarms |", "|---|---|---|---|---|"]
for k, r in res.iterrows():
    b = "**" if k == "Hybrid v2.5" else ""
    rows.append(f"| {b}{k}{b} | {b}{r['Scams caught']}{b} | {b}{r['Alerts correct']:.0%}{b} | {b}{r['F1']:.0%}{b} | "
                f"{b}{r['False alarms']}{b} |")
if gem is None:
    rows.append("| AI only (Gemini) | not run yet (needs GEMINI_API_KEY) | | | |")
lines = ["# Blind test #3: v2.4 vs v2.5 (pre-registered, run once)", "",
         f"Test file: `{TEST}` - {n_s} scams, {n_g} genuine, built by an independent agent from new sources. "
         "Neither version learned from these examples. Event check and website age off for all approaches.", "",
         *rows, "",
         f"Examples where v2.4 and v2.5 disagree: {len(diff)}"]
for _, r in diff.iterrows():
    lines.append(f"- {r['id']} ({'scam' if r['label'] else 'genuine'}): "
                 f"v2.4 {'alert' if r['Hybrid v2.4'] else 'no alert'} (score {r['v24_score']:.2f}), "
                 f"v2.5 {'alert' if r['Hybrid v2.5'] else 'no alert'} (score {r['v25_score']:.2f})")
lines += ["", "Small test set (40), so a difference of 1-2 examples is within noise."]
(OUT / "blind3_report.md").write_text("\n".join(lines), encoding="utf-8")

print(res.drop(columns="recall").to_string(formatters={"Alerts correct": "{:.0%}".format, "F1": "{:.0%}".format}))
print(f"\nv2.4 and v2.5 disagree on {len(diff)} example(s): {', '.join(diff.id) or 'none'}")
print("Saved outputs/blind3_report.md, blind3_chart.png, blind3_hybrid_v24_vs_v25.png, "
      "blind3_results.csv, blind3_predictions.csv")
