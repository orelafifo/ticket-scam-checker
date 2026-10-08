"""
compare_versions.py - v2.2 vs v2.4 on the SAME unseen examples (blind test #2).

Why: a fair before/after needs one test set that neither version learned from.
Blind test #2 (data/blind2_test_set.csv) is that set - v2.2 was built long before it
existed, and v2.4 was frozen (PREREGISTRATION_v24.md) before it was collected.
(The original 80 examples can't be used: v2.4 learns from them.)

Produces two tables with the same four approaches as the first real-world test:
  Table A - v2.2: rules (rules_v22_frozen.py), text model and hybrid trained on the
            v2.2 synthetic data (data/listings_v22.csv, reproduced exactly)
  Table B - v2.4: current rules (features.py), text model on current synthetic data,
            hybrid = the saved v2.4 model
  AI only (Gemini) is identical in both tables (Gemini doesn't change between
  versions). It runs only if GEMINI_API_KEY is in .env; answers are saved in
  outputs/llm_cache_blind2_test_set.json (shared with evaluate_real.py).

Sanity check printed first: v2.2 hybrid on the original 80 examples must give
19/40 caught and 2/40 false alarms - the original v2.2 result.

Run:  python compare_versions.py
Outputs (outputs/):
  compare_v22_vs_v24_report.md   - both tables, ready to paste
  compare_v22_table.png, compare_v24_table.png - one chart per table
  compare_hybrid_v22_vs_v24.png  - the hybrid before vs after
  compare_v22_vs_v24.csv
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import features as v24_rules
import rules_v22_frozen as v22_rules
from channels import channel_info

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
TEST = "data/blind2_test_set.csv"
THRESHOLD = 0.4
WEAK = {"no_seat_details", "high_demand"}


# ---------------------------------------------------------------- helpers
def tri(v):
    return None if v == -1 else bool(v)


def num(v, d):
    return d if pd.isna(v) or v == "" else float(v)


def synth_X(df, rules):
    flags = [rules.all_flags(
        r.text, r.account_age_days, r.followers, r.price_ratio, r.has_seat_details, r.sudden_seller,
        dict(found=tri(r.event_found), details_match=tri(r.event_details_match),
             high_demand=bool(r.event_high_demand), before_on_sale=bool(r.event_before_on_sale),
             app_only=bool(r.event_app_only)),
        dict(category=None if r.ch_category == "none" else r.ch_category, lookalike=bool(r.ch_lookalike),
             new_domain=bool(r.ch_new_domain), insecure_or_short=bool(r.ch_insecure_or_short)))
        for r in df.itertuples()]
    return pd.concat([df[["text"]], pd.DataFrame(flags)], axis=1)


def real_X(df, rules):
    rows = []
    for r in df.itertuples():
        p, f = num(r.price, 0), num(r.face_value, 0)
        ch = channel_info(r.platform, r.link if isinstance(r.link, str) else "", check_age=False)
        rows.append(rules.all_flags(r.text, num(r.account_age_days, 365), num(r.followers, 200),
                                    p / f if p and f else 1.0, int(r.has_seat_details), int(r.sudden_seller),
                                    None, ch))
    return pd.concat([df[["text"]], pd.DataFrame(rows)], axis=1)


def rules_only(X, rules):
    meaningful = [c for c in rules.FLAG_COLUMNS if c not in WEAK and c not in rules.PROTECTIVE]
    return ((X[list(rules.HARD_STOPS)].max(axis=1) == 1) | (X[meaningful].sum(axis=1) >= 2)).astype(int).values


def text_only(train_csv, texts):
    syn = pd.read_csv(train_csv)
    m = Pipeline([("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                  ("c", LogisticRegression(max_iter=1000, class_weight="balanced"))]).fit(syn.text, syn.label)
    return (m.predict_proba(texts)[:, 1] >= THRESHOLD).astype(int)


def hybrid_v22(X):
    """Rebuilds the v2.2 hybrid exactly as it was trained (words + flags, synthetic train split)."""
    syn = pd.read_csv("data/listings_v22.csv")
    S = synth_X(syn, v22_rules)
    Str, _, ytr, _ = train_test_split(S, syn.label, test_size=0.25, stratify=syn.label, random_state=42)
    m = Pipeline([("f", ColumnTransformer([("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2), "text"),
                                           ("x", "passthrough", v22_rules.FLAG_COLUMNS)])),
                  ("c", LogisticRegression(max_iter=1000, class_weight="balanced"))]).fit(Str, ytr)
    hard = (X[list(v22_rules.HARD_STOPS)].max(axis=1) == 1).values
    return ((m.predict_proba(X)[:, 1] >= THRESHOLD) | hard).astype(int)


def hybrid_v24(X):
    model = joblib.load("model/scam_model.joblib")["model"]
    hard = (X[list(v24_rules.HARD_STOPS)].max(axis=1) == 1).values
    return ((model.predict_proba(X)[:, 1] >= THRESHOLD) | hard).astype(int)


# Same prompt and threshold as evaluate_real.py, so the saved answers are shared.
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
        print(f"Gemini incomplete ({len(failed)} failed, often the free-tier limit). Run again later - "
              "saved answers are reused, only missing ones are retried. Gemini row left out for now.")
        return None
    return (np.array(probs) >= 0.5).astype(int)


def metrics(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(y.sum()) - tp; neg = int((1 - y).sum())
    return {"Scams caught": f"{tp}/{int(y.sum())}", "Alerts correct": tp / max(tp + fp, 1),
            "F1": 2 * tp / max(2 * tp + fp + fn, 1), "False alarms": f"{fp}/{neg}",
            "recall": tp / y.sum()}


# ---------------------------------------------------------------- sanity check
dev = pd.read_csv("data/real_test_set.csv")
chk = metrics(dev.label.values, hybrid_v22(real_X(dev, v22_rules)))
ok = chk["Scams caught"] == "19/40" and chk["False alarms"] == "2/40"
print(f"Sanity check - v2.2 hybrid on the original 80 examples: {chk['Scams caught']} caught, "
      f"{chk['False alarms']} false alarms -> {'MATCHES the original v2.2 run' if ok else 'DOES NOT MATCH, tell Claude'}")

# ---------------------------------------------------------------- both tables
test = pd.read_csv(TEST)
y = test.label.values
X22, X24 = real_X(test, v22_rules), real_X(test, v24_rules)
gem = gemini_only(test)

tables = {
    "v2.2": {"Rules only": rules_only(X22, v22_rules),
             "Text model only": text_only("data/listings_v22.csv", test.text),
             "Hybrid v2.2": hybrid_v22(X22)},
    "v2.4": {"Rules only": rules_only(X24, v24_rules),
             "Text model only": text_only("data/listings.csv", test.text),
             "Hybrid v2.4": hybrid_v24(X24)},
}
if gem is not None:
    for t in tables.values():
        t["AI only (Gemini)"] = gem

results = {v: pd.DataFrame({k: metrics(y, p) for k, p in t.items()}).T for v, t in tables.items()}


def md_table(df):
    rows = ["| Approach | Scams caught | Alerts correct | F1 | False alarms |", "|---|---|---|---|---|"]
    for k, r in df.iterrows():
        bold = "**" if k.startswith("Hybrid") else ""
        rows.append(f"| {bold}{k}{bold} | {bold}{r['Scams caught']}{bold} | {bold}{r['Alerts correct']:.0%}{bold} | "
                    f"{bold}{r['F1']:.0%}{bold} | {bold}{r['False alarms']}{bold} |")
    if gem is None:
        rows.append("| AI only (Gemini) | not run yet (needs GEMINI_API_KEY) | | | |")
    return rows


def chart(df, title, file):
    colors = {"Scams caught (recall)": ("recall", "#2a78d6"), "Alerts correct (precision)": ("Alerts correct", "#eb6834"),
              "F1": ("F1", "#1baf7a")}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    xs, w = np.arange(len(df)), 0.26
    for i, (label, (col, colr)) in enumerate(colors.items()):
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
chart(results["v2.2"], f"v2.2 on blind test #2 ({n_s} scams, {n_g} genuine)", "compare_v22_table.png")
chart(results["v2.4"], f"v2.4 on blind test #2 ({n_s} scams, {n_g} genuine)", "compare_v24_table.png")
chart(pd.concat([results["v2.2"].loc[["Hybrid v2.2"]], results["v2.4"].loc[["Hybrid v2.4"]]]),
      f"Hybrid before and after the fix - same examples (blind test #2)", "compare_hybrid_v22_vs_v24.png")

pd.concat({v: r.drop(columns="recall") for v, r in results.items()}).to_csv(OUT / "compare_v22_vs_v24.csv")
lines = ["# v2.2 vs v2.4 on the same unseen examples (blind test #2)", "",
         f"Test file: `{TEST}` - {n_s} scams, {n_g} genuine. Neither version learned from these examples.",
         f"Sanity check: v2.2 hybrid on the original 80 examples = {chk['Scams caught']} caught, "
         f"{chk['False alarms']} false alarms ({'matches' if ok else 'DOES NOT match'} the original v2.2 run).", "",
         "## Table A - v2.2", "", *md_table(results["v2.2"]), "",
         "## Table B - v2.4", "", *md_table(results["v2.4"]), "",
         "Notes: Gemini's row is identical in both tables because Gemini does not change between versions. "
         "Rules only improved because v2.4 uses the wider rulebook. Event check and website age are off for "
         "all approaches. Small test set (40), so treat differences of 1-2 examples with caution."]
(OUT / "compare_v22_vs_v24_report.md").write_text("\n".join(lines), encoding="utf-8")

for v, r in results.items():
    print(f"\nTable {'A' if v == 'v2.2' else 'B'} - {v}")
    print(r.drop(columns="recall").to_string(formatters={"Alerts correct": "{:.0%}".format, "F1": "{:.0%}".format}))
print("\nSaved outputs/compare_v22_vs_v24_report.md, compare_v22_table.png, compare_v24_table.png, "
      "compare_hybrid_v22_vs_v24.png, compare_v22_vs_v24.csv")
