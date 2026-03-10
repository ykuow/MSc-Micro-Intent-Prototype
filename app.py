import streamlit as st
import pandas as pd
from xgboost import XGBClassifier

# --- 1. PROFESSIONAL PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MSc Micro-Intent Engine", 
    page_icon="🛍️", 
    layout="wide"
)

# --- 2. CACHED MODEL LOADING ---
@st.cache_resource
def load_model():
    model = XGBClassifier()
    # Ensure this filename matches your Jupyter output exactly
    model.load_model('xgb_advanced.json')
    return model

try:
    model = load_model()
except Exception as e:
    st.error("Error loading the advanced model. Ensure 'xgb_advanced.json' is in the same folder.")
    st.stop()

# --- 3. HEADER & BRANDING ---
st.title("🛍️ Advanced Micro-Intent Predictor")
# st.markdown("### *Master of Science in Advanced Software Engineering*")
# st.markdown("**University of Westminster, UK | Research Prototype**")
st.divider()

# --- 4. DASHBOARD LAYOUT ---
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.header("👤 Behavioral Metrics")
    st.write("Input real-time session data below:")
    
    # User Input Sliders
    session_len = st.slider("Session Length (Total Clicks)", 1, 100, 20)
    dwell = st.slider("Total Dwell Time (Seconds)", 10, 3600, 600)
    items = st.slider("Unique Items Viewed", 1, 20, 5)
    
    # --- 5. FEATURE ENGINEERING (REPLICATING JUPYTER LOGIC) ---
    # We must match the math used in the training script
    velocity = session_len / (dwell / 60 + 1)
    focus = items / session_len
    
    st.markdown("---")
    st.subheader("Engineered Features")
    st.info(f"**Interaction Velocity:** {velocity:.2f} clicks/min")
    st.info(f"**Focus Index:** {focus:.2f} (lower = more specific)")

with col2:
    st.header("📊 Predictive Intelligence")
    st.write("Click analyze to calculate purchase probability using XGBoost.")
    
    if st.button("RUN ADVANCED PREDICTION", use_container_width=True, type="primary"):
        # Prepare data for prediction
        # Column names must match the order and naming in your Jupyter training X
        input_data = pd.DataFrame(
            [[session_len, dwell, items, velocity, focus]], 
            columns=['session_length', 'total_dwell_time', 'unique_items', 'interaction_velocity', 'focus_index']
        )
        
        # Get Probability and cast to float to prevent Streamlit error
        prob = float(model.predict_proba(input_data)[0][1])
        
        # Display Key Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Purchase Probability", f"{prob*100:.1f}%")
        m2.metric("Model AUC Score", "0.9599")
        m3.metric("Status", "High" if prob > 0.5 else "Low")
        
        # Visual Confidence Progress Bar
        st.write(f"**Prediction Confidence:**")
        st.progress(prob)
        
        st.divider()
        
        # Final Decision Logic
        if prob > 0.5:
            st.success("### ✅ RESULT: HIGH PURCHASE INTENT")
            st.markdown(f"""
            **Analysis:** The model has identified a high-conversion behavioral signature. 
            - **Velocity Profile:** Moderate to High intensity browsing.
            - **Focus Signature:** High interest in specific products.
            - **Confidence:** {prob*100:.2f}%
            """)
            st.balloons()
        else:
            st.warning("### 🧊 RESULT: LOW INTENT / BROWSING")
            st.markdown(f"""
            **Analysis:** Current behavior suggests information gathering or window shopping. 
            - **Velocity Profile:** Low intensity or passive dwell time.
            - **Focus Signature:** Broad, non-specific browsing pattern.
            """)
    else:
        st.info("System Ready. Adjust sliders and execute analysis.")

# --- 6. FOOTER / METHODOLOGY ---
st.divider()
st.caption("Technical Methodology: Behavioral Session Reconstruction (30-min threshold) + Extreme Gradient Boosting (XGBoost). "
           "Trained on RetailRocket E-Commerce Dataset (1.5M+ interactions).")
