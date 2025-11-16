🔋 EV Range Prediction Chatbot

A conversational ML-powered chatbot that predicts electric vehicle driving range based on battery capacity, vehicle weight, speed, and drag coefficient.

🚀 Overview

This project demonstrates how to combine Machine Learning, Streamlit UI, and simple NLP slot-filling to build an interactive chatbot that predicts the real-world driving range of an Electric Vehicle (EV).

You simply chat with the bot using natural language like:

"battery 60 weight 1700 speed 50 cd 0.30"
or
"predict" → and the bot will ask step-by-step questions.

The model uses a Random Forest Regressor trained on an EV specifications dataset.
It includes feature importance, dynamic chat flow, and automatic re-training if a model is missing.

✨ Features
🔹 Interactive Chatbot Interface

Accepts natural text inputs

Extracts parameters automatically

Slot-filling system asks for missing values

Friendly chat-style UI using Streamlit

🔹 Machine Learning-Based Prediction

RandomForestRegressor model

Predicts real-world range (km)

Uses a 4-feature pipeline:

Battery Capacity (kWh)

Curb Weight (kg)

Average Speed (km/h)

Drag Coefficient (Cd)

🔹 Extra Insights

Shows top feature influencing the prediction
