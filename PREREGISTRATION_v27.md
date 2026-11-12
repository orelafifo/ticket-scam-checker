# Pre-registration: v2.7 and blind test #5 (v2.6 vs v2.7)

Written 12 Nov 2026, BEFORE any v2.7 code is written and BEFORE any blind test #5 data exists.
Commit this file before building. Anything below marked "chosen by cross-validation" is chosen
on the training rows only, before blind test #5 is collected.

## Scope
- v2.7 is a checker for **UK live music** resale tickets (concerts and festivals).
- New data (training, library, blind test #5) is UK music only. No Irish or other non-UK rows.
- Existing real rows (240) stay in training, tagged `is_music` / `is_uk`, because scam wording
  transfers across events and removing them would halve the real data. Stated as a limitation.
- Blind test #4 results for v2.6 are reported exactly as run. No rows are removed after the fact.

## Frozen baseline
- v2.6 = git tag `v2.6` + `model/scam_model_v26.joblib` (copy of the v2.6 model).
- v2.6 code is not edited; v2.7 changes happen in new or clearly versioned files.

## Training data for v2.7
- All 240 earlier real rows: real_test_set, heldout_test_set, blind2, blind3, blind4.
  Blind tests #2-#4 have already been run once, so they become training data now.
- Plus new UK music rows collected under DATA_COLLECTION_v27.md (each with a source).
- Synthetic listings stay as background (weight x1); real rows weight x50, as in v2.4-v2.6.

## Upgrade 1: MiniLM meaning score (AI inside the model)
- The decision model stays the v2.6 signed logistic regression on red-flag features
  (expert-tier priors, C = 0.3). Words are NOT added back (v2.4 overfitting lesson).
- ONE new feature, `meaning_score`: a small logistic regression on MiniLM (all-MiniLM-L6-v2)
  embeddings of the normalised text, trained on real rows only. Training rows get
  out-of-fold scores (5-fold) so the main model never sees a score made from its own label.
- `meaning_score` can only raise risk (sign >= 0); prior weight = moderate tier (1.5).
- Its strength (C in 0.01 / 0.1 / 1) is chosen by source-grouped 5-fold CV repeated 5 times.
- Designs compared in that CV: (A) v2.6 design retrained on all real rows; (B) A + meaning_score.
- **B is kept only if** its CV F1 >= A's AND its false-alarm rate is no more than 2 points
  higher. Otherwise v2.7 ships design A and the report says so.
- Paraphrase stress test (data/paraphrase_stress.csv) reported for A and B (v2.6: 28/51, 0/20).

## Upgrade 2: screenshot and conversation input
- Screenshots are read by a local OCR model (EasyOCR; Tesseract if EasyOCR won't install).
  The user sees and can correct the text before checking. Screenshots are not stored.
  OCR text goes through redact.py and the same checker as typed text.
- Conversation mode: a box for the whole chat. New later-stage flags, final list and tiers to
  be confirmed by Orela with sources BEFORE blind test #5 is collected. Draft list:
  `extra_fee` (asks for more money after the first payment: release / transfer / insurance fee),
  `refuses_protection` (refuses an official transfer or protected payment when asked).
- These flags have little or no training data, so their weights stay at their tier (prior).

## Upgrade 3: known-scams library (evidence only)
- Real, sourced UK music scam reports, embedded with MiniLM. The closest match is shown with
  its similarity and source link if similarity >= 0.60.
- **Evidence only: it never changes the score or the risk level.**
- Frozen before blind test #5 is collected. No blind test #5 row can be in it.

## Blind test #5
- Collected by an independent agent that has never seen v2.7 code, results or errors.
- About 20 scams + 20 genuine, UK live music only, new sources not used before.
- At least 5 rows given as screenshots (images) and at least 5 as full conversations.
- Each row labelled with its source before anything is run. "Suspected" rows marked in `notes`
  and results reported with and without them.
- Run ONCE: `python compare_v26_v27.py --test data/blind5_test_set.csv`.

## What will be reported (whatever the result)
- Scams caught, false alarms, precision and F1 for: v2.6, v2.7 model, v2.7 as shown in the app
  (counting "borderline" and "not enough information" as alerts).
- Separately for screenshot rows and conversation rows (v2.6 gets the OCR text, so the
  comparison is about the model, not the input box).
- Every row where v2.6 and v2.7 disagree.
- Main question: does v2.7 catch more scams with no more than 2 extra false alarms out of 20?

## Known limits stated in advance
- About 40 rows: one listing = 5 percentage points; 1-2 listings difference is noise.
- Event check and website-age lookup are off, as in all earlier blind tests.
- "How did you meet the seller" is unknown for test rows unless the source says so.
