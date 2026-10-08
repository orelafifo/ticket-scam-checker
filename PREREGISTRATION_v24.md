# Blind test #2: pre-registration for v2.4 (written BEFORE blind test #2 exists)

Date frozen: 2026-10-08T21:07Z

## Why a second blind test
v2.4 changes the decision model and is trained partly on the 120 real rows from the
development set and blind test #1, so those can no longer judge it fairly.

## What is frozen
- features.py sha256 4353c6ca8f9791a2 (unchanged from v2.3 rules)
- hybrid_model.py sha256 85e642c22ebef76d, step2_train_model.py sha256 af46d7f26237a2bd
- model/scam_model.joblib sha256 6beb2e2fe6757ed5 on the developer machine
- Design chosen by grouped cross-validation only (select_hybrid.py): flags-only signed logistic
  regression, real rows weight x50, threshold 0.4. CV estimate: F1 93%, recall 94%, false alarms 8%.

## Compared (same four as before)
1. Rules only (v2.3 rulebook)  2. Text model only (synthetic, 0.4)  3. AI only (Gemini, 0.5)  4. Hybrid v2.4 (0.4 + hard stops)
Primary result = hybrid v2.4 at 0.4. No secondary thresholds.

## Protocol
New 40-row set built by an independent agent with no access to code, rules, training data or
previous test sets, using sources not used before. Run once, nothing changed afterwards.
