# Ticket Scam Checker (v2 - full Core checks)

## Run it
```
pip install -r requirements.txt
python step1_make_dataset.py     # builds data/listings.csv (synthetic)
python step2_train_model.py      # trains model, prints results, saves charts
streamlit run app.py             # run app
```
## Stop App
```
Cntrl + C                        # stop appp
```
Optional keys (the app works without them):
```
export TICKETMASTER_API_KEY="..."   # live event check (developer.ticketmaster.com)
export GEMINI_API_KEY="..."         # AI explanation (Google AI Studio, free)
```

## Files
| File | Layer |
|---|---|
| features.py | 1. Red-flag rules (20 flags) + hard-stop rules |
| event_check.py | 1b. Event check: Ticketmaster API, falls back to data/events_demo.csv |
| step1_make_dataset.py | Synthetic training data |
| step2_train_model.py | 2. TF-IDF + flags -> logistic regression |
| explainer.py | 3. LLM explanation (Gemini / Claude / template fallback) |
| app.py | Streamlit demo |

## Red flags (Core)
- **Payment:** unsafe_payment, payment_switch, deposit_or_hold, hide_reference
- **Language:** urgency, off_platform, send_after_payment, sob_story, trust_claims, asks_for_codes
- **Seller:** new_account, few_followers, sudden_seller
- **Ticket:** too_cheap, no_seat_details, format_mismatch
- **Event:** event_not_found, event_mismatch, high_demand, before_on_sale

Hard-stop rules (force High risk): asks_for_codes, hide_reference, event_mismatch, format_mismatch.

## Be upfront about (limitations slide)
- Training data is synthetic, and results on it will look too good.
- data/events_demo.csv is made-up DEMO data, not real tour dates.
- The Ticketmaster API mostly covers Ticketmaster-sold events. "Not found" can mean another ticket agent sold it.
- app_only is assumed True for Ticketmaster UK events. Check this per tour.
