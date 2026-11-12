# Data collection for v2.7 (training rows + known-scams library)

These rows are for TRAINING and the LIBRARY only. Blind test #5 is collected separately, later,
by a different agent run, and must never use these sources.

## Rules
- UK live music only (concerts, festivals). No sports, theatre, or non-UK examples.
- Every row has a source: a URL, or "uni group chat (consent given, pseudonymised)".
- Remove names, phone numbers, handles, emails and payment details from the text.
- Label before running the checker on anything. Unsure -> leave it out.
- Do not use any source already in real_test_set, heldout, blind2, blind3 or blind4.
- `source_type`: verbatim (exact quote), reconstructed (built from a description), constructed.
- `evidence_type`: listing, conversation (several messages), or screenshot (image file in
  data/v27_images/, text typed into `text` as well).
- `in_library`: 1 if this scam row should go in the known-scams library (verbatim/reconstructed
  scams from a named source), else 0.

## Where to look
- Scams: Action Fraud and police warnings, Which?, BBC / local news, Citizens Advice,
  bank scam warnings (Lloyds, Barclays, TSB), Reddit (r/LegalAdviceUK, r/unitedkingdom,
  festival and artist subs), Trustpilot reviews of resale sites.
- Genuine: Twickets, Ticketmaster Fan-to-Fan, AXS Official Resale and Facebook / X / Reddit
  posts selling at face value through an official transfer.

## Template
data/v27_new_real_template.csv
