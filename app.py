"""
app.py - the Ticket Scam Checker app (v2.6). Run with:   streamlit run app.py

v2.6: short form (paste the listing, where you found it, how you met the seller); everything
else is optional. The event details are read from the listing by AI (spaCy), soft red flags
can be spotted by meaning (MiniLM), and the result says when there isn't enough
information or the model is unsure, instead of a reassuring "Low".
"""
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

try:  # API keys from a local .env file (never committed - it's in .gitignore)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from channels import PLATFORM_CHOICES, PLATFORMS
from checker import check_listing, load_bundle
from event_check import SELLER_OPTIONS
from explainer import explain
from extract_event import extract_event
from features import CONTACT_CHOICES
from redact import redact

st.set_page_config(page_title="Ticket Scam Checker", page_icon="🎟️", layout="centered")
if not os.path.exists("model/scam_model.joblib"):  # first run online: train the model
    subprocess.run([sys.executable, "step2_train_model.py"], check=True)
bundle = load_bundle()

st.title("🎟️ Ticket Scam Checker")
st.caption("Paste a resale listing before you pay. Prototype (v2.6) – trained on 120 real published scam "
           "cases plus synthetic examples. A warning tool, not a guarantee: always pay with buyer protection.")

# ---------------- The 3 main questions ----------------
text = st.text_area("1. Paste the listing or the seller's messages", height=140,
                    placeholder="e.g. Selling 2x Burna Boy tickets O2 Arena, £40 each, kindly DM...")
platform = st.selectbox("2. Where did you find these tickets?", PLATFORM_CHOICES)
platform = None if platform == PLATFORM_CHOICES[0] else platform
contact_label = st.selectbox("3. How did you come across this seller?", list(CONTACT_CHOICES))
contact = CONTACT_CHOICES[contact_label]

# ---------------- Optional details ----------------
read = extract_event(text) if text.strip() else {"artist": None, "venue": None, "date": None}
with st.expander("Add more details (optional – makes the check more accurate)"):
    link = st.text_input("Link to the listing or website",
                         placeholder="e.g. https://www.viagogo.co.uk/... or a link someone sent you")
    st.caption("We never open the link. We only check the web address itself.")

    price_choice = st.radio("Price", ["Not stated", "Stated in the listing", "Only given in DMs / private messages"],
                            horizontal=True)
    price_status = {"Not stated": "not_stated", "Stated in the listing": "stated",
                    "Only given in DMs / private messages": "dm_only"}[price_choice]
    p1, p2 = st.columns(2)
    price = p1.number_input("Asking price (£ per ticket)", 0.0, 5000.0, 0.0,
                            disabled=price_status != "stated")
    face = p2.number_input("Face value (£ per ticket, 0 = look it up)", 0.0, 5000.0, 0.0)

    st.markdown("**The event** – read from the listing; correct it if it's wrong")
    e1, e2, e3 = st.columns(3)
    artist = e1.text_input("Artist / match", value=read["artist"] or "")
    venue = e2.text_input("Venue", value=read["venue"] or "")
    event_date = e3.date_input("Date", value=read["date"], format="DD/MM/YYYY")
    sellers = st.multiselect("Who sells the OFFICIAL tickets? (check the artist's website; leave empty if "
                             "you don't know)", SELLER_OPTIONS)
    ticket_type = st.selectbox("Ticket type", ["Not stated", "Standing", "Seated – block/row/seat given",
                                               "Seated – no block/row/seat given"])
    seat = ticket_type in ("Standing", "Seated – block/row/seat given")

    st.markdown("**The seller**")
    s1, s2 = st.columns(2)
    age_known = s1.checkbox("I know the account age")
    age = s1.number_input("Seller account age (days)", 0, 10000, 30, disabled=not age_known)
    fol_known = s2.checkbox("I know the follower count")
    followers = s2.number_input("Seller followers", 0, 1000000, 20, disabled=not fol_known)
    sudden = st.checkbox("It's an older account that has suddenly started selling tickets")

if not text.strip():
    pass
elif read["artist"] or read["venue"] or read["date"]:
    found = ", ".join(str(v) for v in [read["artist"], read["venue"], read["date"]] if v)
    st.caption(f"🔎 Event details read from the listing: {found} (edit under 'Add more details').")

clicked = st.button("Check listing", type="primary")
if clicked and not (platform or link.strip()):
    st.warning("Tell us where you found the tickets: pick a platform (or paste the link under 'Add more details').")
elif clicked and not text.strip():
    st.warning("Paste the listing text or the seller's messages.")
elif clicked:
    with st.spinner("Checking the wording, where it's sold and the event..."):
        r = check_listing(text, platform=platform, link=link, contact=contact, price_status=price_status,
                          price=price, face=face, has_seat_details=seat,
                          account_age_days=age if age_known else None,
                          followers=followers if fol_known else None, sudden_seller=sudden,
                          artist=artist, venue=venue, event_date=event_date, official_sellers=sellers,
                          bundle=bundle)
    st.session_state["last"] = dict(r=r, text=text, platform=platform, contact=contact)

if "last" in st.session_state and text.strip() == st.session_state["last"]["text"].strip():
    r = st.session_state["last"]["r"]
    level, score = r["level"], r["score"]
    if r["reason"] == "hard_stop":
        headline = f"### High risk - {'90%+' if score < 0.90 else f'{score:.0%}'}"
    elif r["reason"] == "not_enough_info":
        headline = "### Medium risk - not enough information"
    elif r["reason"] == "borderline":
        headline = f"### Medium risk - borderline ({score:.0%})"
    else:
        headline = f"### {level} risk - {score:.0%}"
    {"High": st.error, "Medium": st.warning, "Low": st.success}[level](headline)
    st.progress(min(max(r["shown"], 0.0), 1.0))
    lo, hi = r["range"]
    st.caption(f"Model score {score:.0%} (likely range {lo:.0%}–{hi:.0%}, from 30 versions of the model "
               "trained on different samples of the data).")
    if r["reason"] == "hard_stop":
        st.caption("Raised to High by a hard-stop rule: " + "; ".join(r["hard_stops"]))
    elif r["reason"] == "borderline":
        st.caption("The model is unsure: some versions of it put this listing above the warning line. "
                   "Treat it with care.")
    elif r["reason"] == "not_enough_info":
        st.caption("This is from social media or a group chat, with no price and nothing known about the seller. "
                   "That's too little to call it safe. Ask for the price, the seat details and proof of purchase, "
                   "and only pay with buyer protection.")

    st.subheader("Red flags found")
    for f in r["red_flags"]:
        st.markdown(f"- 🚩 {f}")
    if not r["red_flags"]:
        st.markdown("No common red flags spotted.")
    for g in r["good_signs"]:
        st.markdown(f"- ✅ {g}")
    if r["ai_evidence"]:
        with st.expander("How the AI spotted some of these"):
            for flag, (sentence, example, sim) in r["ai_evidence"].items():
                st.markdown(f"- **{flag}** – your listing says *\"{sentence}\"*, which means the same as "
                            f"*\"{example}\"* (similarity {sim:.2f})")
            st.caption("MiniLM, a small language model running on this computer, compares the meaning of each "
                       "sentence with example scam phrases. Nothing is sent to an outside service.")

    st.subheader("Event check")
    ev, used = r["event"], r["event_used"]
    if used["artist"]:
        st.markdown(f"Checked: **{used['artist']}**" + (f", {used['venue']}" if used["venue"] else "")
                    + (f", {used['date']:%d %b %Y}" if used["date"] else ""))
    if ev["source"] in ("none", "off"):
        st.markdown(ev["note"] + "  \nThis doesn't change the risk: an event we couldn't check never counts as a red flag.")
    else:
        st.markdown(f"**Source:** {ev['source']}  \n{ev['note']}")
    if ev.get("found"):
        st.markdown(f"Official event: **{ev['official_venue']}**, {ev['official_date']}  \n"
                    f"Status: {ev.get('status') or 'unknown'}"
                    + (f"  \nOfficial price range: £{ev['face_min']:.0f}–£{ev['face_max']:.0f}"
                       if ev.get("face_min") else ""))
    if "Skiddle" in ev.get("source", ""):
        st.caption("Event data from [Skiddle](https://www.skiddle.com).")

    st.subheader("What this means")
    where = platform or (r["channel"].get("link") or {}).get("site") or "a website"
    with st.spinner("Writing explanation..."):
        # v2.6: personal details are removed before the text goes to the AI explainer
        st.write(explain(redact(text), score, level, r["red_flags"], ev["note"], where=where,
                         good_signs=r["good_signs"], channel_category=r["channel"]["category"]))

    # ---------------- Human-in-the-loop: report a wrong result ----------------
    with st.expander("Was this result wrong? Help improve the checker"):
        truth = st.radio("What do you think it really was?", ["Scam", "Genuine", "Not sure"], horizontal=True)
        consent = st.checkbox("I agree to save this listing (personal details removed) so a person can review it")
        if st.button("Send feedback", disabled=not consent):
            log = Path("data/misses_log.csv")
            new = not log.exists()
            with log.open("a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["date", "text_redacted", "platform", "contact", "level", "score", "user_says"])
                w.writerow([datetime.now().isoformat(timespec="minutes"), redact(text),
                            st.session_state["last"]["platform"], st.session_state["last"]["contact"],
                            level, round(score, 3), truth])
            st.success("Thanks - saved for human review. Nothing changes until a person checks it and retrains.")
