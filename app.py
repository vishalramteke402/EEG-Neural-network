import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="EEG Eye State Detection (ANN)",
    layout="centered"
)

st.title("🧠 EEG Eye State Detection")
st.write("Prediction using a **pre-trained Neural Network (ANN) model**")

# ---------------------------------
# Load ANN Model (.pkl)
# ---------------------------------
@st.cache_resource
def load_model():
    with open("cu_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ---------------------------------
# EEG Feature Names
# MUST MATCH TRAINING ORDER
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

    # ANN usually outputs probability
    prob = model.predict(input_df)

    # Handle shape safely
    prob_value = float(prob[0]) if prob.ndim == 1 else float(prob[0][0])
    prediction = 1 if prob_value >= 0.5 else 0

    if prediction == 1:
        st.success("👁️ Eye State: **CLOSED**")
    else:
        st.success("👁️ Eye State: **OPEN**")

    st.subheader("📊 Prediction Probability")
    st.write(f"Closed: **{prob_value*100:.2f}%**")
    st.write(f"Open: **{(1 - prob_value)*100:.2f}%**")
