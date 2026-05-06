import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="centered"
)

@st.cache_resource
def load_model():
    with open("model_ibm.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("✈️ Customer Churn Predictor")
st.markdown("**Predict whether a travel customer will churn using Random Forest (model_ibm.pkl).**")
st.divider()

st.subheader("🧾 Enter Customer Details")
col1, col2 = st.columns(2)

with col1:
    age            = st.slider("Age", 18, 65, 30)
    frequent_flyer = st.selectbox("Frequent Flyer", ["No", "Yes"])
    annual_income  = st.selectbox("Annual Income Class",
                                  ["High Income", "Low Income", "Middle Income"])
with col2:
    services_opted = st.slider("Services Opted", 1, 6, 3)
    account_synced = st.selectbox("Account Synced to Social Media", ["No", "Yes"])
    booked_hotel   = st.selectbox("Booked Hotel", ["No", "Yes"])

# ── Encode exactly as LabelEncoder did during training ──
# FrequentFlyer          : No=0, Yes=1
# AnnualIncomeClass      : High Income=0, Low Income=1, Middle Income=2
# AccountSyncedToSocialMedia : No=0, Yes=1
# BookedHotelOrNot       : No=0, Yes=1
ff_enc     = 1 if frequent_flyer == "Yes" else 0
income_map = {"High Income": 0, "Low Income": 1, "Middle Income": 2}
ai_enc     = income_map[annual_income]
as_enc     = 1 if account_synced == "Yes" else 0
bh_enc     = 1 if booked_hotel   == "Yes" else 0

# Feature order must match model training:
# [Age, FrequentFlyer, AnnualIncomeClass, ServicesOpted,
#  AccountSyncedToSocialMedia, BookedHotelOrNot]
input_data = np.array([[age, ff_enc, ai_enc, services_opted, as_enc, bh_enc]])

st.divider()
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    prediction = model.predict(input_data)[0]
    proba      = model.predict_proba(input_data)[0]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"🚨 **Customer is likely to CHURN** (Confidence: {proba[1]*100:.1f}%)")
        st.markdown("💡 **Recommendation:** Offer personalized discounts or loyalty rewards.")
    else:
        st.success(f"✅ **Customer is NOT likely to churn** (Confidence: {proba[0]*100:.1f}%)")
        st.markdown("💡 **Recommendation:** Continue current engagement strategy.")

    st.metric("Churn Probability", f"{proba[1]*100:.1f}%")
