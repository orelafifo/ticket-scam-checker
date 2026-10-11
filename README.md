# Ticket Scam Checker (v2.5)

Paste a resale ticket listing before you pay: the checker looks at where it's sold, the
wording, the seller and the event, gives a risk level, the red flags it found, and a
plain-English explanation.

## Run it
```
pip install -r requirements.txt
python step1_make_dataset.py     # synthetic background data -> data/listings.csv
python step2_train_model.py      # trains the v2.5 model -> model/scam_model.joblib
streamlit run app.py
```
Optional keys in `.env` (never committed): `TICKETMASTER_API_KEY`, `SKIDDLE_API_KEY`, `GEMINI_API_KEY`.

## Evidence for the report
```
python select_hybrid.py                                   # cross-validation: why v2.4 is designed this way
python evaluate_real.py --test data/blind2_test_set.csv   # v2.4 vs rules / text model / Gemini on unseen examples
python compare_versions.py                                # v2.2 vs v2.4 on the SAME unseen examples (two tables + charts)
python compare_regularisation.py                          # L1 vs L2 penalty and strength comparison (cross-validation)
python select_v25.py                                      # v2.5: chooses the expert-tier strength (cross-validation)
python compare_v24_v25.py                                 # v2.5: blind test #3, v2.4 vs v2.5 (+ Gemini if key set)
```

## Files
| File | Role |
|---|---|
| app.py | Streamlit app |
| channels.py | Where it's sold: platform + link check (never opens the link) |
| features.py | Red-flag rules (34 features incl. 1 good sign), hard-stop rules, v2.5 expert tiers |
| event_check.py | Event check: Ticketmaster + Skiddle, demo catalogue fallback |
| hybrid_model.py | Decision model: sign-constrained logistic regression on the flags. v2.5 pulls each weight towards its expert tier (build_v24 / build_v25) |
| step1_make_dataset.py / step2_train_model.py | Data and training |
| explainer.py | Gemini (or template) explanation - explains, never decides |
| rules_v22_frozen.py, data/listings_v22.csv | Frozen copy of v2.2, used only to rebuild v2.2 for comparison |
| select_hybrid.py / evaluate_real.py / compare_versions.py / compare_regularisation.py / select_v25.py / compare_v24_v25.py | Evaluation scripts |
| PREREGISTRATION_v24.md / PREREGISTRATION_v25.md | What was frozen before blind tests #2 / #3 were collected |
| TIER_SOURCES.md | v2.5: which published source supports each red flag's tier |

## Data
| File | Use |
|---|---|
| data/real_test_set.csv | 80 published-case examples (first real-world test; now training data) |
| data/heldout_test_set.csv | 40 examples, blind test #1 (now training data) |
| data/blind2_test_set.csv | 40 examples, blind test #2 - the fair test for v2.4 (never trained on) |
| data/blind3_test_set.csv | 40 examples, blind test #3 - the fair test for v2.5 (never trained on) |
| data/listings.csv | synthetic background data (v2.4) |
| data/events_demo.csv | demo event catalogue (made-up dates) |

## Version history
- **v2.2** - rules + text model trained on synthetic data. First real-world test: hybrid caught 19/40.
- **v2.4** - overfitting fixed: flag features only, trained mainly on weighted real examples, red flags can
  only raise risk, L2 regularisation, chosen by source-grouped cross-validation, evaluated on a
  pre-registered blind test. On blind test #2: hybrid 17/20 caught, 4/20 false alarms (v2.2 on the same
  examples: 6/20, 0/20).
- **v2.5** - expert evidence tiers: each red flag's weight starts from its tier (hard stop 3.0, strong 2.5,
  moderate 1.5, weak 0.5, good sign -2.0; sources in TIER_SOURCES.md) and training adjusts it (C = 0.3,
  chosen by cross-validation). New flag very_cheap (under 50% of face value). App shows "High risk - 90%+"
  when a hard stop fires, so the number never contradicts the label. On blind test #3: hybrid v2.5 14/20
  caught, 4/20 false alarms; hybrid v2.4 on the same examples 11/20, 2/20.

Hard-stop rules (force High risk): asks_for_codes, hide_reference, event_mismatch, format_mismatch,
lookalike_link, third_party_payment, code_or_access_sale.

Limitations: small test sets; most genuine examples were written rather than collected; synthetic
background data; Gemini may have seen some published scam examples during its training.
