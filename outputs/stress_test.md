# Paraphrase stress test

51 reworded red-flag sentences, 20 ordinary sentences (AI-written test set).

| Version | Red flags spotted | False alarms on ordinary sentences |
|---|---|---|
| v2.5 rules | 8/51 (16%) | 0/20 |
| v2.6 rules | 20/51 (39%) | 0/20 |
| v2.6 + MiniLM | 28/51 (55%) | 0/20 |

## By flag

| Flag | n | v2.5 rules | v2.6 rules | v2.6 + MiniLM |
|---|---|---|---|---|
| deposit_or_hold | 4 | 3 | 3 | 3 |
| off_platform | 15 | 1 | 9 | 10 |
| payment_switch | 3 | 0 | 0 | 2 |
| send_after_payment | 5 | 0 | 0 | 0 |
| sob_story | 5 | 0 | 0 | 1 |
| trust_claims | 4 | 1 | 1 | 2 |
| unsafe_payment | 10 | 2 | 6 | 8 |
| urgency | 5 | 1 | 1 | 2 |

## Still missed by v2.6 + MiniLM

- off_platform: "Ping me privately for more info"
- off_platform: "Get in touch via my Insta and we can chat there"
- off_platform: "Let's continue this conversation over text"
- off_platform: "Contact me outside this group"
- off_platform: "Drop me a line if keen"
- unsafe_payment: "You can just send it straight to my account"
- unsafe_payment: "Transfer to my mum's account is fine"
- send_after_payment: "Once the money lands I'll ping the tickets over"
- send_after_payment: "Pay first then you'll receive them"
- send_after_payment: "Tickets go out to you when I see the transfer"
- send_after_payment: "As soon as you've paid I'll email them"
- send_after_payment: "You'll get them the moment payment clears"
- deposit_or_hold: "Put down a small amount now and the rest on the day"
- urgency: "Need this sorted by tonight"
- urgency: "These won't be around for long"
- urgency: "Going to the first person who sends money"
- sob_story: "My grandma passed away so I won't be going"
- sob_story: "Had to move abroad unexpectedly"
- sob_story: "Broke my leg so can't attend anymore"
- sob_story: "My girlfriend dumped me so I have a spare"
- trust_claims: "Check my profile I've sold loads no issues"
- trust_claims: "You can trust me babe"
- payment_switch: "Card payments keep failing so can we do it another way"

## False alarms (v2.6 + MiniLM)

- none