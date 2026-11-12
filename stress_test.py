"""
stress_test.py - paraphrase stress test: does each version spot a red flag when the
scammer uses DIFFERENT WORDS?

data/paraphrase_stress.csv: 50 red-flag sentences reworded in chat-speak / new wording
(written by Claude, an AI, as an "AI paraphrase" test - disclose this) + 20 ordinary
genuine-listing sentences that should NOT fire. A few rows are the real wordings that
triggered v2.6 (e.g. "Kindly DM if interested"), so they are not all unseen.

Compared:  v2.5 rules   |   v2.6 rules (normalise + new phrases)   |   v2.6 rules + MiniLM
Run:  python stress_test.py      Output: outputs/stress_test.md, outputs/stress_test.csv
"""
from pathlib import Path

import pandas as pd

import features as f26
import rules_v25_frozen as f25
import semantic_flags as sem

SOFT = list(sem.PROTOTYPES)   # flags the meaning-matcher can add
df = pd.read_csv("data/paraphrase_stress.csv")
rows = []
for r in df.itertuples():
    v25 = f25.text_flags(r.text)
    v26r = f26.text_flags(r.text, semantic=False)
    v26a = f26.text_flags(r.text, semantic=True)
    if r.should_fire:
        res = {k: int(v[r.flag] == 1) for k, v in [("v2.5 rules", v25), ("v2.6 rules", v26r), ("v2.6 + MiniLM", v26a)]}
    else:  # a false alarm = any soft red flag fires on an ordinary sentence
        res = {k: int(any(v.get(f) for f in SOFT)) for k, v in
               [("v2.5 rules", v25), ("v2.6 rules", v26r), ("v2.6 + MiniLM", v26a)]}
    rows.append(dict(flag=r.flag, should_fire=r.should_fire, text=r.text, **res,
                     evidence=str(f26.why_semantic(r.text).get(r.flag, "") if r.should_fire else
                                  f26.why_semantic(r.text) or "")))
sem.save_cache()
out = pd.DataFrame(rows)
Path("outputs").mkdir(exist_ok=True)
out.to_csv("outputs/stress_test.csv", index=False)

pos, neg = out[out.should_fire == 1], out[out.should_fire == 0]
cols = ["v2.5 rules", "v2.6 rules", "v2.6 + MiniLM"]
lines = ["# Paraphrase stress test", "",
         f"{len(pos)} reworded red-flag sentences, {len(neg)} ordinary sentences (AI-written test set).", "",
         "| Version | Red flags spotted | False alarms on ordinary sentences |", "|---|---|---|"]
for c in cols:
    lines.append(f"| {c} | {pos[c].sum()}/{len(pos)} ({pos[c].mean():.0%}) | {neg[c].sum()}/{len(neg)} |")
lines += ["", "## By flag", "", "| Flag | n | " + " | ".join(cols) + " |", "|---|---|---|---|---|"]
for flag, g in pos.groupby("flag"):
    lines.append(f"| {flag} | {len(g)} | " + " | ".join(str(g[c].sum()) for c in cols) + " |")
lines += ["", "## Still missed by v2.6 + MiniLM", ""]
lines += [f"- {r.flag}: \"{r.text}\"" for r in pos[pos["v2.6 + MiniLM"] == 0].itertuples()] or ["- none"]
lines += ["", "## False alarms (v2.6 + MiniLM)", ""]
lines += [f"- \"{r.text}\" -> {r.evidence}" for r in neg[neg["v2.6 + MiniLM"] == 1].itertuples()] or ["- none"]
Path("outputs/stress_test.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
