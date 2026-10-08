# v2.2 vs v2.4 on the same unseen examples (blind test #2)

Test file: `data/blind2_test_set.csv` - 20 scams, 20 genuine. Neither version learned from these examples.
Sanity check: v2.2 hybrid on the original 80 examples = 19/40 caught, 2/40 false alarms (matches the original v2.2 run).

## Table A - v2.2

| Approach | Scams caught | Alerts correct | F1 | False alarms |
|---|---|---|---|---|
| Rules only | 14/20 | 78% | 74% | 4/20 |
| Text model only | 20/20 | 61% | 75% | 13/20 |
| **Hybrid v2.2** | **6/20** | **100%** | **46%** | **0/20** |
| AI only (Gemini) | 20/20 | 80% | 89% | 5/20 |

## Table B - v2.4

| Approach | Scams caught | Alerts correct | F1 | False alarms |
|---|---|---|---|---|
| Rules only | 17/20 | 77% | 81% | 5/20 |
| Text model only | 20/20 | 59% | 74% | 14/20 |
| **Hybrid v2.4** | **17/20** | **81%** | **83%** | **4/20** |
| AI only (Gemini) | 20/20 | 80% | 89% | 5/20 |

Notes: Gemini's row is identical in both tables because Gemini does not change between versions. Rules only improved because v2.4 uses the wider rulebook. Event check and website age are off for all approaches. Small test set (40), so treat differences of 1-2 examples with caution.