"""
app.py - the demo app. Run with:   streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

from event_check import check_event
from explainer import explain
from features import FLAG_COLUMNS, all_flags, hard_stops, triggered

st.set_page_config(page_title="Ticket Scam Checker", page_icon="🎟️", layout="centered")
bundle = joblib.load("model/scam_model.joblib")
model, threshold = bundle["model"], bundle["threshold"]

st.title("🎟️ Ticket Scam Checker")
st.caption("Paste a resale listing before you pay. Prototype - synthetic training data.")

# ---------------- 1. The listing ----------------
text = st.text_area("Listing text (paste the post or messages)", height=140,
                    placeholder="e.g. 2x Burna Boy tickets O2 Arena, £40 each, bank transfer only...")

# ---------------- 2. The event ----------------
st.subheader("The event")
e1, e2, e3 = st.columns(3)
artist = e1.text_input("Artist", placeholder="Burna Boy")
venue = e2.text_input("Venue", placeholder="O2 Arena")
event_date = e3.date_input("Date", value=None, format="DD/MM/YYYY")
p1, p2 = st.columns(2)
price = p1.number_input("Asking price (£ per ticket)", 0.0, 5000.0, 40.0)
face = p2.number_input("Face value (£ per ticket, 0 = look it up)", 0.0, 5000.0, 0.0)
seat = st.checkbox("Listing gives section / row / seat details")

# ---------------- 3. The seller ----------------
st.subheader("The seller")
s1, s2 = st.columns(2)
age = s1.number_input("Seller account age (days)", 0, 10000, 30)
followers = s2.number_input("Seller followers", 0, 1000000, 20)
sudden = st.checkbox("It's an older account that has suddenly started selling tickets "
                     "(e.g. a friend's account posting out of character)")

if st.button("Check listing", type="primary") and text.strip():
    event = check_event(artist, venue, event_date)

    # Face value: user's number, else the cheapest official price, else skip the price check
    face_used = face or event.get("face_min") or 0
    price_ratio = price / face_used if face_used else 1.0

    flags = all_flags(text, age, followers, price_ratio, int(seat), int(sudden), event)
    X = pd.DataFrame([{"text": text, **flags}])[["text"] + FLAG_COLUMNS]
    score = float(model.predict_proba(X)[0, 1])
    found = triggered(flags)
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

    st.subheader("Event check")
    st.markdown(f"**Source:** {event['source']}  \n{event['note']}")
    if event.get("found"):
        st.markdown(f"Official event: **{event['official_venue']}**, {event['official_date']}  \n"
                    f"Status: {event.get('status') or 'unknown'}"
                    + (f"  \nOfficial price range: £{event['face_min']:.0f}–£{event['face_max']:.0f}"
                       if event.get("face_min") else ""))
        if event.get("status") in ("cancelled", "postponed", "rescheduled"):
            st.warning(f"This event is {event['status']}. Check the official site before buying anything.")
    if not face_used:
        st.caption("No face value available, so the price check was skipped.")

    st.subheader("What this means")
    with st.spinner("Writing explanation..."):
        st.write(explain(text, score, level, found, event["note"]))
