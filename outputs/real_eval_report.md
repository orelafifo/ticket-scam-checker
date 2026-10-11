# Real-world evaluation

Test file: `data/real_test_set.csv`

Test set: 80 listings (40 scams from published sources: 11 verbatim, 29 reconstructed from documented UK cases; 40 constructed genuine listings). None were used in training.

'Rules only (v2.2 frozen)' uses the rulebook written BEFORE this test set existed; 'Rules only (v2.3)' uses the current rulebook. On the development set, only the frozen version is an untuned result.

| Approach | Scams caught | Recall | Precision | F1 | False alarms |
|---|---|---|---|---|---|
| Rules only (v2.2 frozen) | 30/40 | 75% | 83% | 79% | 6/40 |
| Rules only (v2.3) | 40/40 | 100% | 87% | 93% | 6/40 |
| Text model only | 40/40 | 100% | 63% | 78% | 23/40 |
| Hybrid (our checker) | 35/40 | 88% | 97% | 92% | 1/40 |

For comparison, the hybrid on its own synthetic test set: recall 98%, precision 93%, F1 95%.

Recall by source type:

| Approach | Verbatim quotes | Reconstructed cases |
|---|---|---|
| Rules only (v2.2 frozen) | 64% | 79% |
| Rules only (v2.3) | 100% | 100% |
| Text model only | 100% | 100% |
| Hybrid (our checker) | 91% | 86% |

## Scams the hybrid missed (5)

- **S04** (Facebook scam blog, 2019; score 0.38): "Me and my family can't make it to this wonderful event. Kindly DM me for cheap tickets at a discount rate if you are interested" - flags: social_or_messaging, off_platform, cheap_wording, no_seat_details
- **S15** (Lloyds 2024 (sold-out shows at inflated prices); score 0.32): "Sold out show!! Have 2 VIP standing for Taylor Swift at Wembley, £450 each. Pay upfront to secure, tickets emailed after." - flags: social_or_messaging, deposit_or_hold, send_after_payment, no_seat_details
- **S17** (Lloyds 2026 (football: FA Cup final); score 0.14): "Season ticket holder, can't make the FA Cup final so selling my seat for £120. Need payment today as others are asking." - flags: social_or_messaging, urgency, no_seat_details
- **S22** (GOV.UK/Action Fraud 2025 (last-minute fake ticket companies); score 0.35): "Last minute Glastonbury tickets with coach travel, £395 inc. Limited release, book via our website before they're gone" - flags: unknown_website, urgency, sob_story, no_seat_details
- **S35** (The Music AU (like and comment first); score 0.2): "Comment 'interested' and like this post and I'll message you! 4 tickets for Dua Lipa, reduced price." - flags: social_or_messaging, cheap_wording, engagement_bait, no_seat_details

## Genuine listings the hybrid wrongly flagged (1)

- **G32** (Facebook (post, group or Marketplace); score 0.44): "2 tickets for Sam Fender, St James Park. Paid £85 each, will take £80. Happy to do bank transfer if easier, or paypal G&S, and I'll send via" - flags: social_or_messaging, unsafe_payment, no_seat_details

## Version history on the development set
- v2.2 (before fixes): Rules 30/40 (F1 79%), Text model 38/40 (F1 69%), AI only 39/40 (F1 93%), Hybrid 19/40 (F1 62%, 2 false alarms). Cause: synthetic scams always carried several obvious flags, while real ones are short and vague; several real tactics had no rule.
- v2.3: widened rules (pm me, up front, third-party accounts, gift-card wording, presale-code sales, like-and-comment bait, screenshots as proof, vague posts) and made the training data more realistic (short vague scams, short genuine posts, missing prices). Results above.
- Caution: v2.3 was built after studying this set's mistakes, so results on data/real_test_set.csv are optimistic unless it is a fresh held-out set. The fair test is a new set collected independently (see data/heldout_template.csv).

## Method notes
- Thresholds and the rules-only decision rule were fixed before running the test.
- Event check and website-age lookup were switched off for all approaches (see docstring).
- Unknown seller details were treated as neutral (365-day account, 200 followers).
- Limitations: small test set; genuine listings are constructed, not collected; reconstructed scams were written by the developer, which risks unconscious bias towards the rules. Next step: an independent, real-world labelled set via a bank or platform partner.