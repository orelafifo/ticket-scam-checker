# Pre-registration: blind test #4 (v2.5 vs v2.6)

Written 12 Nov 2026, BEFORE any blind test #4 data exists. Commit this file before collecting the data.

## Frozen before the test (do not change after seeing any result)
- Rules: features.py, normalise.py (slang list), semantic_flags.py (prototypes, anchors,
  SIM_THRESHOLD = 0.70, MARGIN = 0.05)
- Model: model/scam_model.joblib (v2.6, C = 0.3, threshold 0.4, 30-model ensemble)
- Baseline: model/scam_model_v25.joblib + rules_v25_frozen.py (v2.5, unchanged)
- Safety net and borderline rules as in checker.py
- Script: `python compare_v25_v26.py --test data/blind4_test_set.csv`, run ONCE

## The test set (data/blind4_test_set.csv, columns as data/blind4_template.csv)
- Target: about 20 scams + 20 genuine listings, collected from NEW sources not used before
  (not the published cases in real_test_set / heldout / blind2 / blind3).
- Include real listings from uni group chats collected from today onwards, pseudonymised
  (remove names, numbers, handles, links). Label "suspected" ones honestly in a notes column
  and report results with and without them.
- Label each row before running anything. If unsure, leave it out.
- Collected by someone/something that has not seen v2.6's errors (e.g. an independent agent
  run, as in blind test #3), or by Orela without running the checker on the rows first.

## What will be reported (whatever the result)
- Scams caught, false alarms, precision, F1 for: v2.5, v2.6 model, v2.6 as shown in the app
  (counting "Medium - borderline" and "Medium - not enough information" as alerts).
- Every example where v2.5 and v2.6 disagree.
- Main question: does v2.6 catch more scams without more than 2 extra false alarms out of 20?

## Known limits stated in advance
- Test rows don't record "how did you meet the seller", so that new question can't be tested here.
- Event check and website-age lookup are off (as in all earlier blind tests).
- 40 rows is small: a difference of 1-2 listings is within noise.
