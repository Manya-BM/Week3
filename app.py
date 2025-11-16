# app.py
# Streamlit single-file chatbot UI + prediction logic.
# It auto-trains if model.joblib is missing.

import os
import re
import joblib
import numpy as np
import streamlit as st

MODEL_PATH = "model.joblib"
FEATURES = ["battery_kwh", "curb_weight", "avg_speed", "drag_coefficient"]

st.set_page_config(page_title="EV Range Chatbot", layout="centered")
st.markdown("<h1 style='text-align:center'>🔋 EV Range Chatbot</h1>", unsafe_allow_html=True)
st.write(
    "Ask me to predict EV range. Example inputs: `battery 60 weight 1700 speed 50` "
    "or type `predict` to begin."
)

# --- Ensure model exists ---
if not os.path.exists(MODEL_PATH):
    with st.spinner("Training model (first run). This may take ~20-40 seconds..."):
        import ml_train  # assumes ml_train.py is in same folder
        ml_train.train_and_save()

model = joblib.load(MODEL_PATH)


# --- Session state for chat + slots ---
if "history" not in st.session_state:
    st.session_state.history = [
        {
            "role": "bot",
            "text": "Hi — I predict EV range. Type e.g. 'battery 60 weight 1700 speed 50' or say 'predict' to start.",
        }
    ]
if "slots" not in st.session_state:
    st.session_state.slots = {}
if "expecting" not in st.session_state:
    st.session_state.expecting = None


# --- Helpers ---
def extract_slots(text: str) -> dict:
    """
    Robust slot extractor:
    - accepts "battery 60", "battery:60", "60 kwh"
    - accepts "weight 1700", "1700 kg"
    - accepts "speed 50", "50 km/h"
    - accepts "drag 0.30", "cd 0.30"
    - if exactly 4 numbers present and no explicit keywords, map them in order:
      [battery_kwh, curb_weight, avg_speed, drag_coefficient]
    """
    text = text.lower().replace(",", " ")
    slots = {}

    # battery: "battery 60", "60 kwh", "battery:60"
    m = re.search(r"battery\s*[:=]?\s*(\d+(?:\.\d+)?)", text)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)\s*kwh", text)
    if m:
        slots["battery_kwh"] = float(m.group(1))

    # weight: "weight 1700", "1700 kg"
    m = re.search(r"weight\s*[:=]?\s*(\d+(?:\.\d+)?)", text)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)\s*kg", text)
    if m:
        slots["curb_weight"] = float(m.group(1))

    # speed: "speed 50", "50 km/h"
    m = re.search(r"speed\s*[:=]?\s*(\d+(?:\.\d+)?)", text)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)\s*km/?h", text)
    if m:
        slots["avg_speed"] = float(m.group(1))

    # drag coefficient: "drag 0.30", "cd 0.30"
    m = re.search(r"(?:drag|cd|drag_coefficient)\s*[:=]?\s*(\d+(?:\.\d+)?)", text)
    if m:
        slots["drag_coefficient"] = float(m.group(1))

    # Fallback: if at least 4 numbers are present and no explicit slots, map them in order
    if len(slots) < 4:
        nums = re.findall(r"(\d+(?:\.\d+)?)", text)
        nums = [float(n) for n in nums]
        if len(nums) >= 4 and not slots:
            slots = {
                "battery_kwh": nums[0],
                "curb_weight": nums[1],
                "avg_speed": nums[2],
                "drag_coefficient": nums[3],
            }

    return slots


def predict_range(slot_values: dict):
    # ensure feature order and safe conversion
    X = np.array([[float(slot_values.get(f, 0.0)) for f in FEATURES]])
    pred = model.predict(X)[0]
    # feature importance if available
    fi = None
    try:
        rf = model.named_steps.get("rf") or model.named_steps.get("estimator") or model.named_steps.get("model")
        importances = rf.feature_importances_
        norm = (importances / importances.sum()).round(3)
        fi = dict(zip(FEATURES, norm.tolist()))
    except Exception:
        fi = None
    return float(pred), fi


def pretty_slot_name(key: str) -> str:
    return {
        "battery_kwh": "battery capacity (kWh)",
        "curb_weight": "vehicle weight (kg)",
        "avg_speed": "average speed (km/h)",
        "drag_coefficient": "drag coefficient (Cd)",
    }.get(key, key)


# --- Render conversation ---
for msg in st.session_state.history:
    if msg["role"] == "bot":
        st.markdown(
            f"<div style='background:#e6f2ff;padding:8px;border-radius:8px'>{msg['text']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='text-align:right;background:#f0fff0;padding:8px;border-radius:8px'>{msg['text']}</div>",
            unsafe_allow_html=True,
        )


# --- Input form ---
with st.form("msg_form", clear_on_submit=True):
    user_msg = st.text_input("You:", "")
    submitted = st.form_submit_button("Send")

if submitted and user_msg:
    st.session_state.history.append({"role": "user", "text": user_msg})
    lower = user_msg.lower().strip()
    parsed = extract_slots(user_msg)
    # update known slots
    st.session_state.slots.update(parsed)

    if "predict" in lower or parsed or st.session_state.expecting:
        # find missing
        missing = [f for f in FEATURES if f not in st.session_state.slots]
        if missing:
            # if user was previously asked for a particular slot and provided it, clear expecting
            if st.session_state.expecting and st.session_state.expecting in parsed:
                st.session_state.expecting = None
            # ask for next missing slot
            next_slot = missing[0]
            st.session_state.expecting = next_slot
            st.session_state.history.append(
                {"role": "bot", "text": f"Please provide {pretty_slot_name(next_slot)} (example: 60)."}
            )
        else:
            # have all slots — predict
            st.session_state.expecting = None
            try:
                slot_vals = {k: float(st.session_state.slots[k]) for k in FEATURES}
                pred, fi = predict_range(slot_vals)
                explain = f"Predicted range: {pred:.0f} km."
                if fi:
                    top = max(fi.items(), key=lambda kv: kv[1])
                    explain += f" Top influence: {top[0]} (importance {top[1]:.2f})."
                st.session_state.history.append({"role": "bot", "text": explain})
                # visual placeholder
                st.session_state.history.append({"role": "bot", "text": f"__visual__::range::{pred:.1f}"})
            except Exception:
                st.session_state.history.append({"role": "bot", "text": "Error computing prediction. Please check inputs."})
    else:
        # not a prediction intent — smalltalk fallback
        if any(w in lower for w in ["hi", "hello", "hey"]):
            st.session_state.history.append({"role": "bot", "text": "Hello! Ask me to predict EV range."})
        elif "recommend" in lower:
            st.session_state.history.append({"role": "bot", "text": "Recommend feature isn't in this demo. Try 'predict'."})
        else:
            st.session_state.history.append(
                {"role": "bot", "text": "I didn't understand. Try 'predict' or provide values like 'battery 60 weight 1700 speed 50'."}
            )

    # re-run to show updates
    st.rerun()


# After conversation, show any visual placeholders (gauge + feature importances)
for msg in st.session_state.history:
    if msg["role"] == "bot" and msg["text"].startswith("__visual__::range::"):
        try:
            val = float(msg["text"].split("::")[-1])
            st.markdown("---")
            st.write("### Predicted Range")
            st.metric("Predicted range (km)", f"{int(val)} km")
            pct = min(max(val / 700.0, 0.0), 1.0)
            st.progress(pct)
            # feature importances
            try:
                slot_vals = {k: float(st.session_state.slots.get(k, 0.0)) for k in FEATURES}
                _, fi = predict_range(slot_vals)
                if fi:
                    st.write("**Feature importances**")
                    st.table(sorted(fi.items(), key=lambda kv: kv[1], reverse=True))
            except Exception:
                pass
            st.markdown("---")
        except Exception:
            pass
