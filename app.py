"""
app.py - the demo app. Run with:   streamlit run app.py
"""
import os
import subprocess
import sys

import joblib
import pandas as pd
import streamlit as st

# Load API keys from a local .env file (never committed - it's in .gitignore).
# On Streamlit Cloud, keys come from the app's Secrets settings instead.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from channels import PLATFORM_CHOICES, PLATFORMS, channel_info
from event_check import SELLER_OPTIONS, check_event
from explainer import explain
from features import FLAG_COLUMNS, all_flags, hard_stops, protections, triggered

st.set_page_config(page_title="Ticket Scam Checker", page_icon="🎟️", layout="centered")
if not os.path.exists("model/scam_model.joblib"):  # first run online: train the model
    subprocess.run([sys.executable, "step2_train_model.py"], check=True)
bundle = joblib.load("model/scam_model.joblib")
model, threshold = bundle["model"], bundle["threshold"]

st.title("🎟️ Ticket Scam Checker")
st.caption("Paste a resale listing before you pay. Prototype – trained on 120 real published "
           "scam cases plus synthetic examples. A warning tool, not a guarantee: always pay with buyer protection.")

# ---------------- 1. Where are you buying? ----------------
st.subheader("Where are you buying?")
platform = st.selectbox("Where did you find these tickets?", PLATFORM_CHOICES)
platform = None if platform == PLATFORM_CHOICES[0] else platform
link = st.text_input("Link to the listing or website (optional)",
                     placeholder="e.g. https://www.viagogo.co.uk/... or a link someone sent you")
st.caption("We never open the link. We only check the web address itself.")

# ---------------- 2. The listing ----------------
st.subheader("The listing")
text = st.text_area("Listing text (paste the post or messages)", height=140,
                    placeholder="e.g. 2x Burna Boy tickets O2 Arena, £40 each, bank transfer only...")

# ---------------- 3. The event ----------------
st.subheader("The event")
e1, e2, e3 = st.columns(3)
artist = e1.text_input("Artist", placeholder="Burna Boy")
venue = e2.text_input("Venue", placeholder="O2 Arena")
event_date = e3.date_input("Date", value=None, format="DD/MM/YYYY")
sellers = st.multiselect("Who sells the OFFICIAL tickets? Pick all that apply - check the artist's "
                         "website. Leave empty if you don't know.", SELLER_OPTIONS)
p1, p2 = st.columns(2)
price = p1.number_input("Asking price (£ per ticket)", 0.0, 5000.0, 40.0)
face = p2.number_input("Face value (£ per ticket, 0 = look it up)", 0.0, 5000.0, 0.0)
ticket_type = st.selectbox("Ticket type", ["Not stated", "Standing",
                                           "Seated – block/row/seat given",
                                           "Seated – no block/row/seat given"])
# Standing tickets have no seat, so that counts as complete details (no red flag)
seat = ticket_type in ("Standing", "Seated – block/row/seat given")

# ---------------- 4. The seller ----------------
st.subheader("The seller")
s1, s2 = st.columns(2)
age_unknown = s1.checkbox("I don't know the account age")
age = s1.number_input("Seller account age (days)", 0, 10000, 30, disabled=age_unknown)
followers_unknown = s2.checkbox("I don't know the follower count")
followers = s2.number_input("Seller followers", 0, 1000000, 20, disabled=followers_unknown)
# Unknown = neutral (no red flag), the same values used for unknown details in training
if age_unknown:
    age = 365
if followers_unknown:
    followers = 200
sudden = st.checkbox("It's an older account that has suddenly started selling tickets "
                     "(e.g. a friend's account posting out of character)")

clicked = st.button("Check listing", type="primary")
if clicked and not (platform or link.strip()):
    st.warning("First tell us where you're buying: pick a platform or paste the link.")
elif clicked and not text.strip():
    st.warning("Paste the listing text or the seller's messages.")
elif clicked:
    with st.spinner("Checking where it's sold and the event..."):
        channel = channel_info(platform, link)
        event = check_event(artist, venue, event_date, official_sellers=sellers)

    # Face value: user's number, else the cheapest official price, else skip the price check
    face_used = face or event.get("face_min") or 0
    price_ratio = price / face_used if face_used else 1.0

    flags = all_flags(text, age, followers, price_ratio, int(seat), int(sudden), event, channel)
    X = pd.DataFrame([{"text": text, **flags}])[["text"] + FLAG_COLUMNS]
    score = float(model.predict_proba(X)[0, 1])
    found = triggered(flags)
    good = protections(flags)
    stops = hard_stops(flags)

    level = "High" if score >= 0.7 else "Medium" if score >= threshold else "Low"
    if stops:
        level = "High"  # hard-stop rule overrides the model

    msg = f"### {level} risk - model score {score:.0%}"
    {"High": st.error, "Medium": st.warning, "Low": st.success}[level](msg)
    st.progress(score)
    if stops:
        st.caption("Set to High by a hard-stop rule: " + "; ".join(stops))

    st.subheader("Red flags found")
    if found:
        for f in found:
            st.markdown(f"- 🚩 {f}")
    else:
        st.markdown("No common red flags spotted.")
    for g in good:
        st.markdown(f"- ✅ {g}")

    st.subheader("Where it's being sold")
    where = platform or (channel["link"] or {}).get("site") or "a website"
    st.markdown(f"**{where}**" + (f" (link: `{channel['link']['domain']}`)" if channel["link"] else ""))
    if channel["link"]:
        for note in channel["link"]["notes"]:
            st.markdown(f"- {note}")
        if platform and PLATFORMS.get(platform) != channel["link"]["category"] and channel["link"]["site"]:
            st.caption(f"Note: you picked {platform}, but the link goes to {channel['link']['site']}.")

    st.subheader("Event check")
    st.markdown(f"**Source:** {event['source']}  \n{event['note']}")
    if event.get("found"):
        st.markdown(f"Official event: **{event['official_venue']}**, {event['official_date']}  \n"
                    f"Status: {event.get('status') or 'unknown'}"
                    + (f"  \nOfficial price range: £{event['face_min']:.0f}–£{event['face_max']:.0f}"
                       if event.get("face_min") else ""))
        if event.get("status") in ("cancelled", "postponed", "rescheduled"):
            st.warning(f"This event is {event['status']}. Check the official site before buying anything.")
    if "Skiddle" in event["source"]:
        st.caption("Event data from [Skiddle](https://www.skiddle.com).")  # required by Skiddle's API terms
    if not face_used:
        st.caption("No face value available, so the price check was skipped.")

    st.subheader("What this means")
    with st.spinner("Writing explanation..."):
        st.write(explain(text, score, level, found, event["note"], where=where,
                         good_signs=good, channel_category=channel["category"]))
