# Real-world evaluation

Test file: `data/blind2_test_set.csv`

Test set: 40 listings (20 scams from published sources: 4 verbatim, 16 reconstructed from documented UK cases; 20 constructed genuine listings). None were used in training.

'Rules only (v2.2 frozen)' uses the rulebook written BEFORE this test set existed; 'Rules only (v2.4)' uses the current v2.4 rulebook. On the development set, only the frozen version is an untuned result.

| Approach | Scams caught | Recall | Precision | F1 | False alarms |
|---|---|---|---|---|---|
| Rules only (v2.4) | 17/20 | 85% | 77% | 81% | 5/20 |
| Text model only | 20/20 | 100% | 59% | 74% | 14/20 |
| Hybrid (our checker) | 17/20 | 85% | 81% | 83% | 4/20 |

For comparison, the hybrid on its own synthetic test set: recall 98%, precision 78%, F1 87%.

Recall by source type:

| Approach | Verbatim quotes | Reconstructed cases |
|---|---|---|
| Rules only (v2.4) | 75% | 88% |
| Text model only | 100% | 100% |
| Hybrid (our checker) | 50% | 94% |

## Scams the hybrid missed (3)

- **B13** (eFestivals forum 2011; score 0.18): "I HAVE 4 TICKETS FOR SALE FOR V FESTIVAL WITH WEEKEND CAMPING. I HAVE GOT THE TICKETS IN HAND I PAID AROUND £195 EACH PER TICKETS. UNABLE TO" - flags: social_or_messaging, too_cheap, no_seat_details
- **B22** (Louder / AI Incident Database 2024; score 0.18): "Hi folks, I hope you're all well out there. Backstage tickets for my next show in your cities are now going for only $800, which were previo" - flags: social_or_messaging, no_seat_details
- **B23** (Liverpool FC 2023; score 0.18): "[Post in a closed Snapchat ticketing group titled 'No Scammers'] Tickets for Liverpool home game at Anfield available, message me." - flags: social_or_messaging, no_seat_details

## Genuine listings the hybrid wrongly flagged (4)

- **B09** (Instagram; score 0.75): "selling 1 x reading festival weekend camping, can't go anymore (resit exams 🙃). £300 which is under what i paid (£314 w fees). happy to put " - flags: social_or_messaging, off_platform, sob_story, no_seat_details
- **B14** (X (Twitter); score 0.41): "selling 2 standing for sabrina carpenter o2 tues night bc ive got covid 😭 face value £87.65 each, listed them on ticketmaster resale so its " - flags: social_or_messaging, sob_story, no_seat_details
- **B15** (Other official seller (Eventim, Gigantic...); score 0.55): "Ticket Exchange - Arsenal v Everton, Premier League. Block 23, Row 14, Seat 512. Adult £64.00. Available to members with purchase history; t" - flags: unknown_website
- **B24** (Gumtree or other classified ads; score 0.66): "For sale: 2 adult weekend EOTR tickets, face value £130 each. 1st person to offer £260 for the two gets them, I'm not looking to make any pr" - flags: social_or_messaging, screenshot_proof, no_seat_details

## Held-out protocol
- The checker, model, rules, thresholds and Gemini prompt were frozen and recorded in the PREREGISTRATION file BEFORE this set existed.
- The set was built by an independent agent that had not seen the code, rules, training data or development set. It was run once; nothing was changed afterwards.
- Pre-declared secondary result, hybrid at threshold 0.30: scams caught 17/20, false alarms 4/20, F1 83%.

## Method notes
- Thresholds and the rules-only decision rule were fixed before running the test.
- Event check and website-age lookup were switched off for all approaches (see docstring).
- Unknown seller details were treated as neutral (365-day account, 200 followers).
- Limitations: small test set; genuine listings are constructed, not collected; reconstructed scams were written by the developer, which risks unconscious bias towards the rules. Next step: an independent, real-world labelled set via a bank or platform partner.