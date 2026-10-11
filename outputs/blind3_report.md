# Blind test #3: v2.4 vs v2.5 (pre-registered, run once)

Test file: `data/blind3_test_set.csv` - 20 scams, 20 genuine, built by an independent agent from new sources. Neither version learned from these examples. Event check and website age off for all approaches.

| Approach | Scams caught | Alerts correct | F1 | False alarms |
|---|---|---|---|---|
| Rules only | 14/20 | 74% | 72% | 5/20 |
| Text model only | 20/20 | 62% | 77% | 12/20 |
| Hybrid v2.4 | 11/20 | 85% | 67% | 2/20 |
| **Hybrid v2.5** | **14/20** | **78%** | **74%** | **4/20** |
| AI only (Gemini) | not run yet (needs GEMINI_API_KEY) | | | |

Examples where v2.4 and v2.5 disagree: 5
- C17 (scam): v2.4 no alert (score 0.37), v2.5 alert (score 0.40)
- C22 (genuine): v2.4 no alert (score 0.37), v2.5 alert (score 0.40)
- C27 (genuine): v2.4 no alert (score 0.37), v2.5 alert (score 0.40)
- C32 (scam): v2.4 no alert (score 0.37), v2.5 alert (score 0.40)
- C34 (scam): v2.4 no alert (score 0.37), v2.5 alert (score 0.40)

Small test set (40), so a difference of 1-2 examples is within noise.