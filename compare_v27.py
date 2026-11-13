"""
compare_v27.py - blind test #5 (pre-registered in PREREGISTRATION_v27.md, Amendment 1). Run ONCE:

    python compare_v27.py --test data/blind5_test_set.csv

Approaches, all on the SAME new examples (event check and website age off for all):
  1. Rules only        v2.7 rulebook: any hard stop, or 2+ meaningful red flags
  2. Text model only   TF-IDF + logistic regression on the synthetic listings, threshold 0.4
  3. AI only (Gemini)  asked directly "is this a scam?", threshold 0.5 (needs GEMINI_API_KEY;
                       answers saved in outputs/llm_cache_<test>.json so a re-run never re-asks)
  4. v2.5              frozen rules + frozen model
  5. v2.6              frozen model (model/scam_model_v26.joblib)
  6. v2.7 model        frozen model (model/scam_model_v27.joblib): red flags + MiniLM meaning score
  7. v2.7 app view     also counts "Medium - borderline" and "Medium - not enough information"

Outputs (outputs/):
  blind5_report.md                - tables to paste + every example where versions disagree
  blind5_chart.png                - rules only vs text only vs Gemini only vs v2.7
  blind5_v25_vs_v27.png           - v2.5 vs v2.7
  blind5_v26_vs_v27.png           - v2.6 vs v2.7 (the pre-registered main question)
  blind5_results.csv, blind5_predictions.csv
"""
from __future__ import annotations

import argparse
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

import features as f27
import rules_v25_frozen as f25
import semantic_flags as sem
from data_utils import real_flags

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--test", default="data/blind5_test_set.csv")
TEST = ap.parse_args().test
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
THRESHOLD = 0.4
WEAK = {"no_seat_details", "high_demand"}

v25 = joblib.load("model/scam_model_v25.joblib")
v26 = joblib.load("model/scam_model_v26.joblib")
v27 = joblib.load("model/scam_model_v27.joblib")
assert v26.get("version") == "2.6" and v27.get("version") == "2.7", "frozen model files are wrong"

test = pd.read_csv(TEST)
y = test.label.values
texts = test.text.astype(str).values
suspected = test.get("notes", pd.Series([""] * len(test))).fillna("").str.lower().str.contains("suspect").values
price_status = np.where(test.price.isna(), "not_stated", "stated")
seller_known = ~(test.account_age_days.isna() & test.followers.isna()).values

# ---------------------------------------------------------------- flags
X25 = real_flags(test, f25.all_flags)
X27 = pd.DataFrame([real_flags(test.iloc[[i]], f27.all_flags, price_status=price_status[i]).iloc[0]
                    for i in range(len(test))])[f27.FLAG_COLUMNS]   # v2.6 and v2.7 share this rulebook
hard25 = (X25[list(f25.HARD_STOPS)].max(axis=1) == 1).values
hard27 = (X27[list(f27.HARD_STOPS)].max(axis=1) == 1).values

# ---------------------------------------------------------------- the approaches
preds, scores = {}, {}

meaningful = [c for c in f27.FLAG_COLUMNS if c not in WEAK and c not in f27.PROTECTIVE]
preds["Rules only"] = (hard27 | (X27[meaningful].sum(axis=1) >= 2).values).astype(int)

syn = pd.read_csv("data/listings.csv")
text_model = Pipeline([("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                       ("c", LogisticRegression(max_iter=1000, class_weight="balanced"))]).fit(syn.text, syn.label)
scores["Text model only"] = text_model.predict_proba(texts)[:, 1]
preds["Text model only"] = (scores["Text model only"] >= THRESHOLD).astype(int)

LLM_PROMPT = """You are checking a resale ticket listing for fraud.
Where it is being sold: {platform}{link}
Listing / messages:
\"\"\"{text}\"\"\"
Reply with JSON only: {{"scam_probability": <number 0 to 1>, "reason": "<max 15 words>"}}"""


def gemini_only():
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
                continue
            cache[r.id] = answer
            cache_file.write_text(json.dumps(cache, indent=1))
        probs.append(float(cache[r.id]["scam_probability"]))
    if failed:
        print(f"Gemini incomplete ({len(failed)} failed: {', '.join(failed)}). Run again later - "
              "saved answers are reused, nothing else changes.")
        return None
    return np.array(probs)


g = gemini_only()
if g is not None:
    scores["AI only (Gemini)"] = g
    preds["AI only (Gemini)"] = (g >= 0.5).astype(int)

scores["v2.5"] = v25["model"].predict_proba(X25)[:, 1]
preds["v2.5"] = ((scores["v2.5"] >= THRESHOLD) | hard25).astype(int)

scores["v2.6"] = v26["model"].predict_proba(X27)[:, 1]
preds["v2.6"] = ((scores["v2.6"] >= THRESHOLD) | hard27).astype(int)

X27m = X27.copy()
X27m["meaning_score"] = v27["meaning"].score(texts)
scores["v2.7"] = v27["model"].predict_proba(X27m)[:, 1]
preds["v2.7"] = ((scores["v2.7"] >= THRESHOLD) | hard27).astype(int)
lo, hi = v27["ensemble"].interval(X27m)
net = np.array([f27.not_enough_info(X27.iloc[i].to_dict(), price_status[i], bool(seller_known[i]), None)
                for i in range(len(test))])
preds["v2.7 (app view)"] = (preds["v2.7"].astype(bool) | (hi >= THRESHOLD) | net).astype(int)
sem.save_cache()


def metrics(yy, p):
    tp = int(((p == 1) & (yy == 1)).sum()); fp = int(((p == 1) & (yy == 0)).sum())
    fn = int(yy.sum()) - tp; neg = int((1 - yy).sum())
    return {"Scams caught": f"{tp}/{int(yy.sum())}", "recall": tp / max(yy.sum(), 1),
            "Alerts correct": tp / max(tp + fp, 1), "F1": 2 * tp / max(2 * tp + fp + fn, 1),
            "False alarms": f"{fp}/{neg}", "fa_rate": fp / max(neg, 1)}


res = pd.DataFrame({k: metrics(y, p) for k, p in preds.items()}).T
keep = ~suspected
res_ns = pd.DataFrame({k: metrics(y[keep], p[keep]) for k, p in preds.items()}).T


def chart(df, title, file):
    cols = {"Scams caught (recall)": ("recall", "#2a78d6"), "Alerts correct (precision)": ("Alerts correct", "#eb6834"),
            "F1": ("F1", "#1baf7a")}
    fig, ax = plt.subplots(figsize=(max(6, 1.9 * len(df) + 1.5), 4.4))
    xs, w = np.arange(len(df)), 0.26
    for i, (label, (col, colr)) in enumerate(cols.items()):
        vals = df[col].astype(float)
        bars = ax.bar(xs + (i - 1) * w, vals, w - 0.02, color=colr, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=8, color="#333333")
    ax.set_xticks(xs, [f"{i}\nfalse alarms {df.loc[i, 'False alarms']}" for i in df.index], fontsize=8)
    ax.set_ylim(0, 1.2); ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=10, loc="left")
    plt.tight_layout(); plt.savefig(OUT / file, dpi=200); plt.close()


n_s, n_g = int(y.sum()), int((1 - y).sum())
main = [k for k in ["Rules only", "Text model only", "AI only (Gemini)", "v2.7"] if k in res.index]
chart(res.loc[main], f"Blind test #5 ({n_s} scams, {n_g} genuine): v2.7 vs simpler approaches", "blind5_chart.png")
chart(res.loc[["v2.5", "v2.7"]], "v2.5 vs v2.7 - same unseen examples (blind test #5)", "blind5_v25_vs_v27.png")
chart(res.loc[["v2.6", "v2.7"]], "v2.6 vs v2.7 - same unseen examples (blind test #5)", "blind5_v26_vs_v27.png")
res.drop(columns=["recall", "fa_rate"]).to_csv(OUT / "blind5_results.csv")

per = test[["id", "label", "evidence_type", "platform", "source"]].copy() if "evidence_type" in test else \
      test[["id", "label", "platform", "source"]].copy()
per["suspected"] = suspected.astype(int)
for k, s in scores.items():
    per[f"{k} score"] = np.round(s, 2)
for k, p in preds.items():
    per[k] = p
per.to_csv(OUT / "blind5_predictions.csv", index=False)


def table(df):
    rows = ["| Approach | Scams caught | Alerts correct | F1 | False alarms |", "|---|---|---|---|---|"]
    for k, r in df.iterrows():
        b = "**" if k == "v2.7" else ""
        rows.append(f"| {b}{k}{b} | {b}{r['Scams caught']}{b} | {b}{r['Alerts correct']:.0%}{b} | "
                    f"{b}{r['F1']:.0%}{b} | {b}{r['False alarms']}{b} |")
    return rows


def answer(base):
    b, n = res.loc[base], res.loc["v2.7"]
    more = int(n["Scams caught"].split("/")[0]) - int(b["Scams caught"].split("/")[0])
    extra_fa = int(n["False alarms"].split("/")[0]) - int(b["False alarms"].split("/")[0])
    ok = more > 0 and extra_fa <= 2
    return (f"v2.7 vs {base}: {more:+d} scams caught, {extra_fa:+d} false alarms -> "
            f"**{'YES' if ok else 'NO'}** (needs more scams caught and no more than 2 extra false alarms).")


def disagreements(a, b):
    d = per[per[a] != per[b]]
    out = [f"Examples where {a} and {b} disagree: {len(d)}"]
    for _, r in d.iterrows():
        out.append(f"- {r['id']} ({'scam' if r['label'] else 'genuine'}{', suspected' if r['suspected'] else ''}): "
                   f"{a} {'alert' if r[a] else 'no alert'} ({r[f'{a} score']:.2f}), "
                   f"{b} {'alert' if r[b] else 'no alert'} ({r[f'{b} score']:.2f})")
    return out


lines = ["# Blind test #5 (pre-registered, run once)", "",
         f"Test file: `{TEST}` - {n_s} scams, {n_g} genuine, collected by an independent agent from new "
         "UK live music sources. No version learned from these examples. Event check and website age off.", "",
         "## All approaches", "", *table(res), ""]
if g is None:
    lines += ["AI only (Gemini): not run (needs GEMINI_API_KEY in .env).", ""]
lines += ["## Pre-registered questions", "", "- " + answer("v2.6"), "- " + answer("v2.5"), "",
          f"## Without the {int(suspected.sum())} suspected scam row(s)", "", *table(res_ns), ""]
if "evidence_type" in test:
    conv = (test.evidence_type == "conversation").values
    if conv.any():
        res_c = pd.DataFrame({k: metrics(y[conv], p[conv]) for k, p in preds.items()}).T
        lines += [f"## Conversation rows only ({int(conv.sum())})", "", *table(res_c), ""]
lines += ["## Where versions disagree", "", *disagreements("v2.6", "v2.7"), "", *disagreements("v2.5", "v2.7"), "",
          "Small test set: one example = 5 percentage points; a difference of 1-2 examples is within noise."]
(OUT / "blind5_report.md").write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print("\nSaved outputs/blind5_report.md, blind5_chart.png, blind5_v25_vs_v27.png, blind5_v26_vs_v27.png, "
      "blind5_results.csv, blind5_predictions.csv")
