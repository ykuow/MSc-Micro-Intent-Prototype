import streamlit as st
import pandas as pd
from xgboost import XGBClassifier

# Load the saved model
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model('xgb_baseline.json')
    return model

model = load_model()

st.title("🛍️ Micro-Intent Purchase Predictor")
st.markdown("This prototype predicts if an anonymous user will make a purchase based on their micro-behaviors.")

# User Inputs
st.sidebar.header("Mock Session Inputs")
session_length = st.sidebar.slider("Session Length (Total Clicks)", 2, 50, 5)
avg_dwell_time = st.sidebar.slider("Average Dwell Time (Milliseconds)", 1000, 120000, 15000)

if st.button("Predict Purchase Intent"):
    # Format input for model
    input_data = pd.DataFrame({'session_length': [session_length], 'avg_dwell_time': [avg_dwell_time]})
    
    # Predict
    prediction = model.predict(input_data)
    # Grab the probability of class 1 (Buy)
    probability = model.predict_proba(input_data)[0][1] 
    
    st.subheader("Prediction Results:")
    if prediction[0] == 1:
        st.success(f"🛒 HIGH INTENT: User is likely to Buy! (Probability: {probability*100:.2f}%)")
    else:
        st.warning(f"👁️ LOW INTENT: User is just browsing. (Probability: {probability*100:.2f}%)")