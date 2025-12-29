import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(
    page_title="EEG Eye State Detection (ANN)",
    layout="centered"
)

st.title("🧠 EEG Eye State Detection")
st.write("Real-time prediction using a **trained Neural Network (ANN)**")

# ---------------------------------
# Load ANN model
# ---------------------------------
@st.cache_resource
def load_ann_model():
    return load_model("ann_eeg_model.h5")

model = load_ann_model()

# ---------------------------------
# Load Scaler
# ---------------------------------
@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

scaler = load_scaler()

# ---------------------------------
# EEG Feature Names (ORDER MATTERS!)
# ---------------------------------
feature_names = [
    'AF3','F7','F3','FC5','T7','P7','O1',
    'O2','P8','T8','FC6','F4','F8','AF4'
]

# ---------------------------------
# Sidebar Inputs
# ---------------------------------
st.sidebar.header("🔧 EEG Channel Inputs")

user_input = {}
for feature in feature_names:
    user_input[feature] = st.sidebar.number_input(
        label=feature,
        value=0.0,
        format="%.4f"
    )

input_df = pd.DataFrame([user_input])

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("🔍 Predict Eye State"):

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict probability
    prob = model.predict(input_scaled)
    prob_value = float(prob[0][0])

    # Classification
    prediction = "CLOSED" if prob_value >= 0.5 else "OPEN"

    # Output
    st.success(f"👁️ Eye State: **{prediction}**")

    st.subheader("📊 Prediction Probability")
    st.write(f"Closed: **{prob_value * 100:.2f}%**")
    st.write(f"Open: **{(1 - prob_value) * 100:.2f}%**")
