# Blind test #3: pre-registration for v2.5 (written BEFORE blind test #3 exists)

Date frozen: 2026-10-09T15:02Z

## Why a third blind test
v2.5 changes how the red-flag weights are learned. Each weight now starts from an expert evidence tier (see TIER_SOURCES.md) and training adjusts it. v2.5 also adds one flag, very_cheap (price under 50% of face value). The prior strength (C = 0.3) was chosen by looking at cross-validation results. Blind test #2 has already been seen while v2.5 was being designed, so a fresh set is needed to judge v2.5 fairly.

## What is frozen (sha256, first 16 characters)
- features.py: 36d0702df1d473f3 (34 flags, tiers, PRIOR_WEIGHTS)
- hybrid_model.py: 36d3bc53570a9822 (V25_PRIOR_C = 0.3)
- step2_train_model.py: 23e7381aa3cf3ef1
- channels.py: 9fae9ed5f039801d
- model/scam_model.joblib: 5c998484c28dbce1 (on Claude's build machine; a retrain on another machine can give a different file hash but the same weights)
- Design chosen by grouped cross-validation only (select_v25.py). CV estimate for v2.5:
  - F1: 93.0%
  - recall: 95%
  - false alarms: 9%
  - (v2.4 for comparison: F1 92.9%, recall 94%, false alarms 8%)

## Compared on the same 40 examples
1. Rules only (v2.5 rulebook: any hard stop, or 2+ meaningful red flags)
2. Text model only (TF-IDF on synthetic listings, threshold 0.4)
3. AI only (Gemini, zero-shot, threshold 0.5): same prompt as before; runs on the developer's machine with their key
4. **Hybrid v2.4** (rebuilt exactly with `build_v24()`, threshold 0.4 + hard stops)
5. **Hybrid v2.5** (the saved model, threshold 0.4 + hard stops)

**Primary result:** Hybrid v2.5 vs Hybrid v2.4, compared on F1, scams caught and false alarms.

**Expectation, stated now:** cross-validation shows the two as equal (93.0% vs 92.9%), so a big difference is NOT expected. v2.5's main benefit is weights that make sense and match the evidence, plus a score display that doesn't contradict the label; it is not mainly about accuracy. If v2.5 does worse than v2.4 by 2 or more examples, that will be reported as a real weakness.

## Protocol
- A new 40-row set (IDs C01–C40): 20 scams and 20 genuine.
- It is built by an independent agent with no access to the code, rules, tiers, training data or previous test sets.
- It uses only sources not used in any earlier set (all earlier URLs are excluded).
- It is run ONCE with `python compare_v24_v25.py`.
- Nothing (rules, tiers, C, threshold, prompt) is changed after seeing the results. Whatever comes out is reported.
- Event check and website-age lookup are off for every approach, as in the earlier tests.
- Limitation, stated now: with the event check off, the event flags (event_mismatch, event_not_found, before_on_sale) can't fire. So this test can't judge their new tier weights either.
