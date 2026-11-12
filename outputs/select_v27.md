# v2.7 Upgrade 1: does the meaning score help?

291 real rows (142 scam, 149 genuine) + 750 synthetic. Source-grouped 5-fold cross-validation, repeated 5 times. Threshold 0.4 + hard stops.

| Design | Meaning-model C | F1 | Scams caught | Precision | False alarms |
|---|---|---|---|---|---|
| A: v2.6 design, all real rows | - | 84.7% | 84% | 85% | 14% |
| B: A + meaning score | 0.01 | 84.9% | 84% | 86% | 13% |
| B: A + meaning score | 0.1 | 85.1% | 84% | 86% | 13% |
| B: A + meaning score | 1.0 | 91.1% | 93% | 89% | 11% |

Best B: C = 1.0. Pre-registered rule: keep B only if F1 >= A (91.1% vs 84.7%) and false alarms are no more than 2 points higher (11% vs 14%).

**Decision: KEEP the meaning score (design B).**

## The 51 new UK music rows (v2.6 never saw them)

| Model | Scams caught | False alarms | F1 |
|---|---|---|---|
| v2.6 frozen | 12/22 | 1/29 | 69% |
| v2.7 chosen design (cross-validated) | 17/22 | 1/29 | 85% |

## Paraphrase sentences (AI-written): average meaning score

- Reworded red-flag sentences: 0.69 (46/51 at 0.5 or above)
- Ordinary listing sentences: 0.54 (10/20 at 0.5 or above)

Limits: most new scam rows were reconstructed from descriptions, not quoted; the 51-row check is cross-validated, not a blind test. Blind test #5 is the real test.