"""
event_check.py - Layer 1b: check the listing against the real event.

Answers four Core questions:
  1. Does this event exist?                  -> found
  2. Do the date and venue match?            -> details_match
  3. Is it a sold-out / high-demand show?    -> high_demand
  4. Is it being sold before general sale?   -> before_on_sale
It also returns app_only (for the PDF/screenshot check), the event status
(cancelled / postponed) and the official face-value range.

Sources (keys go in a .env file, see .env.example):
  * LIVE:    Ticketmaster Discovery API (free key: developer.ticketmaster.com)
             TICKETMASTER_API_KEY=...   (also covers Live Nation events)
  * LIVE:    Skiddle Events API (free key: skiddle.com/api/join.php)
             SKIDDLE_API_KEY=...        (UK gigs, clubs, festivals)
  * NOT POSSIBLE: AXS, See Tickets, DICE, Eventim, Gigantic have no public API.
             For these the user picks the official seller, and "not found" is
             treated as "couldn't check" rather than a red flag.
  * OFFLINE: data/events_demo.csv - a small DEMO catalogue so the app works on
             camera without internet. It is made-up sample data, NOT real tour
             dates. Say so in the video.

Values are True / False / None. None = "couldn't check", which never raises a
red flag by itself (we don't punish a seller because our lookup failed).

Limitation to mention: no single source covers every UK ticket agent, so
"not found" only counts as a red flag when the official seller is one we search.
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

CATALOGUE = Path(__file__).parent / "data" / "events_demo.csv"
TM_URL = "https://app.ticketmaster.com/discovery/v2/events.json"
SKIDDLE_URL = "https://www.skiddle.com/api/v1/events/search/"
GENERIC_VENUE_WORDS = {"the", "arena", "stadium", "london", "manchester", "birmingham", "live", "hall"}


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def venue_matches(a: str, b: str) -> bool:
    """'O2 Arena' should match 'The O2'. Share a meaningful word, or look similar."""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    key_a = set(a.split()) - GENERIC_VENUE_WORDS
    key_b = set(b.split()) - GENERIC_VENUE_WORDS
    if key_a & key_b:
        return True
    return SequenceMatcher(None, a, b).ratio() > 0.75


def artist_matches(query: str, name: str) -> bool:
    return _norm(query) in _norm(name) or SequenceMatcher(None, _norm(query), _norm(name)).ratio() > 0.85


def _empty(source: str, note: str) -> dict:
    return dict(found=None, details_match=None, high_demand=False, before_on_sale=False,
                app_only=None, status=None, face_min=None, face_max=None,
                official_date=None, official_venue=None, source=source, note=note)


def _judge(events: list[dict], venue: str, event_date: date | None, today: date, source: str) -> dict:
    """events: list of dicts with keys date, venue, on_sale, status, high_demand, app_only, face_min, face_max."""
    if not events:
        r = _empty(source, "No official event found for this artist.")
        r["found"] = False
        return r

    def is_match(e):
        ok_venue = venue_matches(venue, e["venue"]) if venue else True
        ok_date = (e["date"] == event_date) if event_date else True
        return ok_venue and ok_date

    matches = [e for e in events if is_match(e)]
    # show the closest official event: same venue first, then nearest date
    best = matches[0] if matches else min(
        events, key=lambda e: (not (venue and venue_matches(venue, e["venue"])),
                               abs((e["date"] - (event_date or today)).days) if e["date"] else 9999))

    on_sale = best.get("on_sale")
    return dict(
        found=True,
        details_match=bool(matches),
        high_demand=bool(best.get("high_demand")),
        before_on_sale=bool(on_sale and today < on_sale),
        app_only=best.get("app_only"),
        status=best.get("status"),
        face_min=best.get("face_min"),
        face_max=best.get("face_max"),
        official_date=best["date"].isoformat() if best.get("date") else None,
        official_venue=best.get("venue"),
        source=source,
        note="Matched an official event." if matches else
             "Artist is touring, but not on this date / at this venue.",
    )


# ---------------------------------------------------------------------------
# Source 1: Ticketmaster Discovery API
# ---------------------------------------------------------------------------
def _ticketmaster(artist: str, api_key: str) -> list[dict]:
    import requests
    resp = requests.get(TM_URL, timeout=6, params={
        "apikey": api_key, "keyword": artist, "countryCode": "GB",
        "classificationName": "music", "size": 100, "sort": "date,asc"})
    resp.raise_for_status()
    events = []
    for e in resp.json().get("_embedded", {}).get("events", []):
        names = [e.get("name", "")] + [a.get("name", "") for a in e.get("_embedded", {}).get("attractions", [])]
        if not any(artist_matches(artist, n) for n in names):
            continue
        start = e.get("dates", {}).get("start", {}).get("localDate")
        status = e.get("dates", {}).get("status", {}).get("code")      # onsale / offsale / cancelled / postponed
        on_sale = e.get("sales", {}).get("public", {}).get("startDateTime")
        prices = e.get("priceRanges") or [{}]
        event_day = date.fromisoformat(start) if start else None
        events.append(dict(
            date=event_day,
            venue=(e.get("_embedded", {}).get("venues") or [{}])[0].get("name", ""),
            on_sale=datetime.fromisoformat(on_sale.replace("Z", "+00:00")).date() if on_sale else None,
            status=status,
            # 'offsale' before the show has happened usually means sold out
            high_demand=(status == "offsale" and event_day is not None and event_day >= date.today()),
            # Assumption: UK Ticketmaster arena/stadium shows use app (SafeTix) tickets. Verify per tour.
            app_only=True,
            face_min=prices[0].get("min"), face_max=prices[0].get("max"),
        ))
    return events


# ---------------------------------------------------------------------------
# Source 2: Skiddle Events API (UK gigs, clubs, festivals)
# Free key: apply at https://www.skiddle.com/api/join.php
# Terms: you must credit Skiddle (name + logo) wherever its data is shown.
# ---------------------------------------------------------------------------
def _skiddle(artist: str, api_key: str, event_date: date | None) -> list[dict]:
    import requests
    params = {"api_key": api_key, "keyword": artist, "limit": 100, "description": 1}
    if event_date:  # narrow the search to the month around the date
        params["minDate"] = date(event_date.year, event_date.month, 1).isoformat()
    resp = requests.get(SKIDDLE_URL, timeout=6, params=params)
    resp.raise_for_status()
    events = []
    for e in resp.json().get("results", []) or []:
        names = [e.get("eventname", "")] + [a.get("name", "") for a in (e.get("artists") or [])]
        if not any(artist_matches(artist, n) for n in names):
            continue
        day = e.get("date") or (e.get("startdate") or "")[:10]
        price = str(e.get("entryprice") or "").replace("£", "").split()[0] if e.get("entryprice") else None
        try:
            price = float(price) if price else None
        except ValueError:
            price = None
        events.append(dict(
            date=date.fromisoformat(day) if day else None,
            venue=(e.get("venue") or {}).get("name", ""),
            on_sale=None,                      # Skiddle doesn't give a general on-sale date
            status="cancelled" if str(e.get("cancelled", "0")) == "1" else "onsale",
            high_demand=False,                 # not available from Skiddle
            app_only=None,                     # unknown: Skiddle often uses e-tickets, so don't assume
            face_min=price, face_max=price,
        ))
    return events


# ---------------------------------------------------------------------------
# Source 3: offline demo catalogue
# ---------------------------------------------------------------------------
def _catalogue(artist: str) -> list[dict] | None:
    """Returns None if the artist isn't in the demo catalogue at all
    (a small demo list can't prove an event doesn't exist)."""
    if not CATALOGUE.exists():
        return None
    df = pd.read_csv(CATALOGUE)
    df = df[df["artist"].apply(lambda n: artist_matches(artist, n))]
    if df.empty:
        return None
    return [dict(date=date.fromisoformat(r.date), venue=r.venue,
                 on_sale=date.fromisoformat(r.on_sale_date), status=r.status,
                 high_demand=bool(r.high_demand), app_only=bool(r.app_only),
                 face_min=r.face_min, face_max=r.face_max) for r in df.itertuples()]


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
# Ticket sellers we can search. AXS, See Tickets, DICE, Eventim and Gigantic
# have no public API (partner access only), so we can't confirm their events.
SEARCHABLE_SELLERS = {"Ticketmaster", "Skiddle"}
SELLER_OPTIONS = ["Don't know", "Ticketmaster", "Skiddle", "AXS", "See Tickets",
                  "DICE", "Eventim", "Gigantic", "Other"]


def check_event(artist: str, venue: str = "", event_date: date | None = None,
                today: date | None = None, official_seller: str = "Don't know") -> dict:
    """official_seller = who sells the official tickets (from the artist's website).
    We only say 'event not found' when that seller is one we can actually search;
    otherwise a missing result just means 'couldn't check' (no red flag)."""
    today = today or date.today()
    if not artist or not artist.strip():
        return _empty("none", "No artist entered, so the event wasn't checked.")

    # 1) Ask every live source we have a key for
    sources = {"Ticketmaster": ("TICKETMASTER_API_KEY", lambda k: _ticketmaster(artist, k)),
               "Skiddle": ("SKIDDLE_API_KEY", lambda k: _skiddle(artist, k, event_date))}
    events, searched, problems, hits = [], [], [], []
    for name, (env_var, fetch) in sources.items():
        key = os.getenv(env_var)
        if not key:
            continue
        try:
            found = fetch(key)
            searched.append(name)
            events += found
            if found:
                hits.append(name)
        except Exception as e:  # network down, bad key, rate limit -> carry on, never crash
            problems.append(f"{name} lookup failed ({e.__class__.__name__})")

    if events:
        return _judge(events, venue, event_date, today, " + ".join(hits))

    if searched:
        if official_seller in SEARCHABLE_SELLERS and official_seller in searched:
            r = _empty(" + ".join(searched), f"No official {official_seller} event found for this artist.")
            r["found"] = False
            return r
        elif official_seller in SEARCHABLE_SELLERS:  # its key is missing or its lookup failed
            problems.append(f"Not found on {' or '.join(searched)}, and {official_seller} couldn't be searched")
        else:
            seller = "the official seller" if official_seller == "Don't know" else official_seller
            # don't return yet: the demo catalogue below may still know the artist
            problems.append(f"Not found on {' or '.join(searched)}. Tickets may be sold by {seller}, "
                            "which we can't search, so check the artist's official website")

    # 2) Fall back to the offline demo catalogue
    reason = "; ".join(problems) or "No API keys set, used demo catalogue"
    events = _catalogue(artist)
    if events is None:
        return _empty("demo catalogue", reason + ". Event not checked.")
    result = _judge(events, venue, event_date, today, "demo catalogue")
    result["note"] = f"{result['note']} ({reason})"
    return result


if __name__ == "__main__":
    # quick manual test:  python event_check.py
    for a, v, d in [("Burna Boy", "O2 Arena", date(2027, 3, 14)),
                    ("Burna Boy", "O2 Arena", date(2027, 3, 20)),
                    ("Stray Kids", "Wembley", date(2027, 7, 3)),
                    ("Unknown Band", "", None)]:
        print(a, v, d, "->", check_event(a, v, d))
