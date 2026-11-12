"""
checker.py - v2.6: one function that runs the whole check, so the app, the tests and the
command-line demo all use exactly the same logic.

Order of steps
  1. Where it's sold (channels.py)
  2. Event details: what the buyer typed, otherwise read from the listing (extract_event.py, spaCy)
  3. Event check (event_check.py) - optional
  4. Red flags (features.py: normalised text + rules + MiniLM meaning-matcher)
  5. Decision model score + ensemble range (hybrid_model.py)
  6. Risk level, in this order:
       hard stop                        -> High (shown as 90%+)
       score >= 0.7                     -> High
       score >= 0.4                     -> Medium
       ensemble range crosses 0.4       -> Medium "borderline - the model is unsure"
       not enough information (safety net) -> Medium "not enough information"
       otherwise                        -> Low
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from channels import channel_info
from event_check import check_event
from extract_event import extract_event
from features import (FLAG_COLUMNS, all_flags, hard_stops, not_enough_info, protections, triggered,
                      why_semantic, ALL_DESCRIPTIONS)

NEUTRAL_AGE, NEUTRAL_FOLLOWERS = 365, 200   # unknown seller details = neutral (as in training)


@lru_cache(maxsize=1)
def load_bundle(path="model/scam_model.joblib"):
    return joblib.load(path)


def check_listing(text, platform=None, link="", contact=None, price_status=None, price=0.0, face=0.0,
                  has_seat_details=False, account_age_days=None, followers=None, sudden_seller=False,
                  artist="", venue="", event_date=None, official_sellers=None, run_event_check=True,
                  bundle=None) -> dict:
    bundle = bundle or load_bundle()
    model, threshold = bundle["model"], bundle["threshold"]

    channel = channel_info(platform, link or "")

    # Event details: typed by the buyer, else read from the listing
    read = extract_event(text)
    artist_used = artist or read["artist"] or ""
    venue_used = venue or read["venue"] or ""
    date_used = event_date or read["date"]
    if run_event_check:
        event = check_event(artist_used, venue_used, date_used, official_sellers=official_sellers)
    else:
        event = {"found": None, "details_match": None, "source": "off", "note": "Event check switched off."}

    face_used = face or event.get("face_min") or 0
    price_ratio = price / face_used if (price and face_used and price_status in (None, "stated")) else 1.0
    seller_known = account_age_days is not None or followers is not None

    flags = all_flags(text, NEUTRAL_AGE if account_age_days is None else account_age_days,
                      NEUTRAL_FOLLOWERS if followers is None else followers, price_ratio,
                      int(bool(has_seat_details)), int(bool(sudden_seller)), event, channel,
                      contact=contact, price_status=price_status)
    X = pd.DataFrame([flags])[FLAG_COLUMNS]
    score = float(model.predict_proba(X)[0, 1])
    lo, hi = (float(v[0]) for v in bundle["ensemble"].interval(X)) if "ensemble" in bundle else (score, score)

    stops = hard_stops(flags)
    reason = None
    if stops:
        level = "High"; reason = "hard_stop"
    elif score >= 0.7:
        level = "High"
    elif score >= threshold:
        level = "Medium"
    elif hi >= threshold:
        level = "Medium"; reason = "borderline"
    elif not_enough_info(flags, price_status, seller_known, contact):
        level = "Medium"; reason = "not_enough_info"
    else:
        level = "Low"
    shown = max(score, 0.90) if stops else score

    ai = why_semantic(text)
    return dict(score=score, shown=shown, range=(lo, hi), level=level, reason=reason, threshold=threshold,
                flags=flags, red_flags=triggered(flags), good_signs=protections(flags), hard_stops=stops,
                ai_evidence={ALL_DESCRIPTIONS[k]: v for k, v in ai.items()},
                event=event, event_read=read, event_used=dict(artist=artist_used, venue=venue_used, date=date_used),
                channel=channel, face_used=face_used)


if __name__ == "__main__":
    # The real suspected scam from a uni WhatsApp group chat (Nov 2026), pseudonymised.
    WA = ("Hello, I'm selling 3x Olivia Rodrigo Tickets The Unraveled Tour @ The O2 Arena, London on "
          "Friday 9th April 2027. Kindly Dm if interested.")
    v25 = joblib.load("model/scam_model_v25.joblib")
    import rules_v25_frozen as f25
    from channels import channel_info as ci
    f = f25.all_flags(WA, 365, 200, 1.0, 0, 0, None, ci("WhatsApp", ""))
    print(f"v2.5 (as in the app, nothing else filled in): {v25['model'].predict_proba(pd.DataFrame([f]))[0, 1]:.0%}")
    for kw in [dict(), dict(contact="group_chat", price_status="not_stated")]:
        r = check_listing(WA, platform="WhatsApp", run_event_check=False, **kw)
        print(f"v2.6 {kw or '(short form, questions left blank)'}: {r['level']} - score {r['score']:.0%} "
              f"(range {r['range'][0]:.0%}-{r['range'][1]:.0%}) reason={r['reason']}")
        print("   red flags:", r["red_flags"])
        print("   AI evidence:", r["ai_evidence"])
        print("   event read from text:", r["event_read"])
