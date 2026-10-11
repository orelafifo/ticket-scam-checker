# Held-out test: pre-registration (written BEFORE the held-out set exists)

Date frozen: 2026-10-08T20:50Z

## What is frozen
- Code version v2.3: features.py (sha256 4353c6ca8f9791a2), channels.py, step1/step2 as in repo.
- Model: retrained from step1 v2.3 data, seed 42 (model sha256 3b387cb30908f849 on the developer machine).
- Gemini prompt for "AI only": the LLM_PROMPT in evaluate_real.py, threshold 0.5.

## The four approaches compared (same as the development-set chart)
1. Rules only (v2.3 rulebook): scam if any hard-stop rule or 2+ meaningful red flags.
2. Text model only: TF-IDF + logistic regression on the synthetic listings, threshold 0.4.
3. AI only (Gemini, zero-shot), threshold 0.5.
4. Hybrid (the checker): rules + text in one model, threshold 0.4, plus hard-stop rules.
   Secondary, declared now: hybrid at threshold 0.30 (suggested by the development set). Reported separately, not used to pick a winner.

## Protocol
- The held-out set is built by an independent agent that has NOT seen the code, rules, training data or development set.
- It is run ONCE. No rules, thresholds, prompts or data are changed after seeing it. Whatever comes out is reported.
- Event check and website-age lookup are off for all approaches (as on the development set).
