# Ticket Scam Checker (v2.7 - in progress)

For UK live music (concerts and festivals).

Paste a resale ticket listing before you pay. The checker looks at the wording, where it's sold,
how you met the seller, the seller and the event. It gives a risk level, the red flags it found
(and how the AI spotted them), and a plain-English explanation.

## Run it
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # spaCy's English model (once)
python step2_train_v27.py                 # trains v2.7 (first run downloads MiniLM, ~90 MB, needs internet once)
streamlit run app.py
python -m pytest -q                       # 54 unit tests
```
Optional keys in `.env` (never committed): `TICKETMASTER_API_KEY`, `SKIDDLE_API_KEY`, `GEMINI_API_KEY`.
If MiniLM or spaCy can't load, the app still works with the rules alone and says so.

## What's new in v2.7 (in progress - see PREREGISTRATION_v27.md)
- **Scope:** UK live music only.
- **More real training data:** 291 real examples (blind tests #2-#4 + 51 new UK music examples),
  up from 120. Every example now has a `notes` column saying whether it is a real message,
  rebuilt from a news story, or written for the project.
- **Upgrade 1 - AI meaning score (done):** MiniLM reads the whole message and a small model
  learned from the real examples how much it sounds like known scams. That one score is an extra
  input to the decision model (not shown to users). Kept because it passed the pre-registered rule
  in cross-validation: F1 84.7% -> 91.1%, false alarms 14% -> 11% (`select_v27.py`).
- **Next:** Upgrade 2 (reading screenshots + whole conversations), Upgrade 3 (library of known
  scams, shown as evidence only), then blind test #5.
- v2.6 is frozen as the baseline: git tag `v2.6` + `model/scam_model_v26.joblib`.

## What's new in v2.6: AI inside the checker (not just calling an LLM)
| Part | AI technique | File |
|---|---|---|
| Text normalisation | Slang dictionary + fuzzy matching ("hmu" -> "hit me up", "plsease" -> "please") | normalise.py |
| Meaning-matcher | MiniLM sentence embeddings, run locally: a sentence fires a soft red flag if it MEANS the same as an example scam phrase | semantic_flags.py |
| Event details from the text | spaCy named-entity recognition + venue/artist ruler ("Olivia Rodrigo", "The O2", 9 Apr 2027) | extract_event.py |
| "The model is unsure" | Bootstrap ensemble: 30 copies of the model trained on resampled data; if they disagree about the warning line -> "Medium - borderline" | hybrid_model.py, step2_train_model.py |
| Decision | Same expert-tier logistic regression as v2.5 (it still decides; the AI helps it see) | hybrid_model.py |
| Explanation | Gemini explains only, now after personal details are removed | explainer.py, redact.py |

Also new: the question "How did you come across this seller?" (3 new flags), price
stated / not stated / only in DMs, a safety net (social or group-chat listing + no price + no
seller details = "Medium - not enough information", never Low), a short form, an opt-in
"Was this wrong?" log for human review (personal details removed). The made-up demo event
catalogue is no longer used by the app: with no API key, no internet or no result, the event
is shown as "not checked" (never a red flag). The file is kept only for the unit tests.

Principle: AI reads and suggests; rules and the model decide; a human approves changes.

## Evidence
```
python select_v26.py        # cross-validation: v2.5 vs v2.6 rules vs v2.6 + MiniLM; picks the similarity threshold
python stress_test.py       # paraphrase stress test: does each version spot red flags in new wording?
python compare_v25_v26.py   # v2.5 vs v2.6 on blind tests #2 and #3 (development check - NOT a fair test)
python compare_v25_v26.py --test data/blind4_test_set.csv   # blind test #4, pre-registered, run once
python checker.py           # the WhatsApp case: v2.5 vs v2.6
python select_v27.py        # v2.7 Upgrade 1: with vs without the meaning score (cross-validation)
```

## Files
| File | Role |
|---|---|
| app.py | Streamlit app (short form + optional details) |
| checker.py | The whole check in one function (used by the app and tests) |
| normalise.py / semantic_flags.py / extract_event.py | v2.6 AI parts (see table above) |
| features.py | Red-flag rules (37 features incl. 1 good sign), hard stops, expert tiers, safety net |
| channels.py / event_check.py | Where it's sold / event check (Ticketmaster, Skiddle; "not checked" if unavailable) |
| hybrid_model.py | Decision model (expert-tier logistic regression) + ensemble; build_v27 adds the meaning score |
| meaning_model.py | v2.7 MiniLM meaning score |
| step2_train_v27.py | Trains the v2.7 model the app uses |
| step1_make_dataset.py / step2_train_model.py | Data and training |
| explainer.py / redact.py | Gemini (or template) explanation / personal-detail removal |
| rules_v25_frozen.py, model/scam_model_v25.joblib | Frozen v2.5, used only for comparison |
| data_utils.py | Shared data helpers |
| select_v26.py / stress_test.py / compare_v25_v26.py | Evaluation scripts |
| tests/test_v26.py, tests/test_v27.py | Unit tests |
| PREREGISTRATION_v27.md, DATA_COLLECTION_v27.md | v2.7 rules, set before building |
| model/scam_model_v26.joblib | Frozen v2.6, used only for comparison |
| PREREGISTRATION_v26.md | What is frozen before blind test #4 |
| TIER_SOURCES.md | Sources for each red flag's tier (v2.6 additions at the end - to confirm) |

## Data
| File | Use |
|---|---|
| data/real_test_set.csv, data/heldout_test_set.csv | 120 published-case rows (training) |
| data/blind2_test_set.csv, data/blind3_test_set.csv, data/blind4_test_set.csv | earlier blind tests - training data for v2.7 |
| data/v27_new_real_rows.csv | 51 new UK music examples (v2.7 training) |
| data/v27_orela_rows.csv | Orela's own examples - waiting, not yet used for training |
| data/used_sources_v27.txt | sources already used, so new data never repeats them |
| data/paraphrase_stress.csv | 51 reworded red-flag sentences + 20 ordinary ones (AI-written) |
| data/blind4_template.csv | column layout for blind test #4 |
| data/listings.csv | synthetic background data |
| data/events_test_catalogue.csv | made-up events, used ONLY by the unit tests (never by the app) |
| data/misses_log.csv | created by the app when a user reports a wrong result (opt-in, redacted) |

## Version history
- **v2.2** - rules + text model on synthetic data. First real-world test: hybrid caught 19/40.
- **v2.4** - overfitting fixed (flag features only, real examples weighted, sign constraints, L2,
  grouped CV, pre-registered blind test). Blind test #2: 17/20 caught, 4/20 false alarms.
- **v2.5** - expert evidence tiers as informative priors (C = 0.3), very_cheap flag, "High risk - 90%+".
  Blind test #3: 14/20 caught, 4/20 false alarms.
- **v2.6** - AI inside the checker: text normalisation, MiniLM meaning-matcher, spaCy event
  extraction, ensemble uncertainty; "how did you meet the seller" question, price status, safety
  net, short form, misses log. Blind test #4: 12/20 caught, 0/20 false alarms (app view 15/20, 5/20).
- **v2.7** (in progress) - UK live music scope, 291 real training examples, MiniLM meaning score.

Limitations: small test sets; only 75 of 291 real training examples are real word-for-word messages
(114 scams rebuilt from news descriptions, 102 genuine examples written for the project); synthetic
background data; the 3 new contact flags have no training data yet (weights = expert tier);
MiniLM struggles with negation ("I'll transfer before you pay" can look like "after you pay").
