import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load models and encoders
model_condition = joblib.load("model_condition_xgb.pkl")
model_risk = joblib.load("model_risk_xgb.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder_condition = joblib.load("label_encoder_condition.pkl")
label_encoder_risk = joblib.load("label_encoder_risk.pkl")

# Ensure session state
if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="Stampede Risk Predictor", layout="centered")
st.title("🚨 Human Stampede Risk Predictor (Condition + Risk Model)")
st.markdown("Predict stampede risk using *Crowd Condition* and *Stampede Risk* models.")

# ---- Input Form ----
with st.form("stampede_prediction_form"):
    number_of_deaths = st.number_input("🢍 Number of Deaths", min_value=0, max_value=10000, value=0)
    country_code = st.selectbox("🌐 Country Code (e.g., India = 7)", list(range(0, 20)), index=7)
    estimated_capacity = st.number_input("🏟️ Estimated Actual Capacity", min_value=1, max_value=10000000, value=50000)
    arrived_capacity = st.number_input("🛫 Arrived Capacity", min_value=0, max_value=10000000, value=120000)

    submitted = st.form_submit_button("🔍 Predict Crowd Risk")

if submitted:
    # Input preparation
    density_ratio = round(arrived_capacity / estimated_capacity, 2)

    input_data = pd.DataFrame([{
        "Number of Deaths": number_of_deaths,
        "Country_Code": country_code,
        "Estimated Actual Capacity": estimated_capacity,
        "Arrived Capacity": arrived_capacity,
        "Crowd Density": density_ratio
    }])

    input_scaled = scaler.transform(input_data)

    # === Model Predictions ===
    cond_pred = model_condition.predict(input_scaled)[0]
    cond_proba = model_condition.predict_proba(input_scaled)[0]
    cond_label = label_encoder_condition.inverse_transform([cond_pred])[0]

    risk_pred = model_risk.predict(input_scaled)[0]
    risk_proba = model_risk.predict_proba(input_scaled)[0]
    risk_label = label_encoder_risk.inverse_transform([risk_pred])[0]

    # Correct class labels
    cond_classes_model = model_condition.classes_
    cond_class_labels = label_encoder_condition.inverse_transform(cond_classes_model)
    cond_proba_dict = dict(zip(cond_class_labels, cond_proba.round(3)))

    risk_classes_model = model_risk.classes_
    risk_class_labels = label_encoder_risk.inverse_transform(risk_classes_model)
    risk_proba_dict = dict(zip(risk_class_labels, risk_proba.round(3)))

    # Interpret Density
    if density_ratio <= 1:
        density_condition = "Comfortable"
    elif density_ratio <= 2:
        density_condition = "Manageable"
    elif density_ratio <= 3:
        density_condition = "Tightly Packed (risk increases)"
    else:
        density_condition = "Critical (stampede likely)"

    # Record Result
    result_record = {
        "Number of Deaths": number_of_deaths,
        "Country_Code": country_code,
        "Estimated Capacity": estimated_capacity,
        "Arrived Capacity": arrived_capacity,
        "Crowd Density": density_ratio,
        "Crowd Condition": cond_label,
        "Stampede Risk": risk_label,
        "Density Condition": density_condition
    }
    st.session_state.history.append(pd.DataFrame([result_record]))

    # Display Results
    st.subheader("📊 Prediction Results")

    st.markdown("### 🔷 Crowd Condition Model")
    st.success(f"**Predicted Crowd Condition:** `{cond_label}`")
    fig1, ax1 = plt.subplots()
    ax1.pie(cond_proba, labels=cond_class_labels, autopct="%1.1f%%", startangle=90, colors=plt.cm.Paired.colors)
    ax1.axis("equal")
    st.pyplot(fig1)
    st.write("🔍 Class Probabilities", pd.DataFrame([cond_proba_dict]))

    st.markdown("### 🔺 Stampede Risk Model")
    st.success(f"**Predicted Stampede Risk:** `{risk_label}`")
    fig2, ax2 = plt.subplots()
    ax2.pie(risk_proba, labels=risk_class_labels, autopct="%1.1f%%", startangle=90, colors=plt.cm.Set3.colors)
    ax2.axis("equal")
    st.pyplot(fig2)
    st.write("🔍 Class Probabilities", pd.DataFrame([risk_proba_dict]))

    st.markdown("### 🧾 Input Summary")
    st.write(pd.DataFrame([result_record]))

# ---- History ----
if st.session_state.history:
    st.markdown("---")
    st.subheader("📂 Prediction History")
    history_df = pd.concat(st.session_state.history, ignore_index=True)
    st.dataframe(history_df)

    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button("📅 Download Prediction History as CSV", data=csv, file_name="prediction_history.csv", mime="text/csv")
