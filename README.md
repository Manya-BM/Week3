# EV Range Chatbot (Streamlit demo)

Simple single-machine demo of an EV prediction chatbot using Streamlit and scikit-learn.

## What it is
- A chat-style UI that predicts real-world EV range from 4 inputs:
  - battery_kwh (kWh)
  - curb_weight (kg)
  - avg_speed (km/h)
  - drag_coefficient (Cd)
- If you don't have a dataset, the app trains on a synthetic dataset so it works immediately.

## Setup (VS Code)
1. Open folder in VS Code.
2. Create and activate a virtual environment:
   - Windows:
     ```
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - macOS / Linux:
     ```
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Install dependencies:
