import streamlit as st
import pandas as pd
import joblib
import datetime

# =============================
# 📦 Load Dataset and Model
# =============================
@st.cache_data
def load_data():
    return pd.read_csv("C:\Users\Data Professor\Desktop\MTN Project\mtn_customer_churn.csv")
    
@st.cache_resource
def load_data():
    return pd.read_csv("mtn_customer_churn.csv")

data = load_data()
model = load_model()

# =============================
# 🎨 App Config and Styling
# =============================
st.set_page_config(page_title="MTN Customer Retention Predictor", page_icon="📶", layout="centered")

# --- Dark Theme with MTN Yellow ---
st.markdown("""
    <style>
        .stApp {
            background-color: #121212;
            color: #f0f0f0;
        }
        h1, h2, h3, h4, h5, h6, label, .css-10trblm {
            color: #ffcc00 !important;
        }
        .highlight {
            color: #ffcc00;
            font-weight: bold;
        }
        .stButton > button {
            background-color: #ffcc00;
            color: #000000;
            border-radius: 8px;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #ffd633;
            color: #000000;
        }
        .stSelectbox div, .stSlider div, .stNumberInput div {
            color: #f0f0f0;
        }
    </style>
""", unsafe_allow_html=True)

# =============================
# 🚀 Header
# =============================
st.markdown("# 📶 <span class='highlight'>MTN Customer Retention Predictor</span>", unsafe_allow_html=True)
st.markdown("### 🧠 Predict whether a customer will renew their MTN subscription.")
st.markdown("💬 _Fill in the customer details to estimate renewal likelihood._")

# =============================
# 📝 Input Form
# =============================
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        date_of_purchase = st.date_input("📅 Date of Purchase", value=datetime.date.today())
        age = st.number_input("🎂 Age", min_value=10, max_value=100)
        state = st.selectbox("📍 State", sorted(data["State"].dropna().unique()))
        mtn_device = st.selectbox("📱 MTN Device", sorted(data["MTN Device"].dropna().unique()))
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
    
    with col2:
        satisfaction_rate = st.slider("⭐ Satisfaction Rate (1–5)", 1, 5, 3)
        customer_review = st.selectbox("🗣️ Customer Review", sorted(data["Customer Review"].dropna().unique()))
        customer_tenure = st.number_input("📊 Customer Tenure (Months)", min_value=0)
        subscription_plan = st.selectbox("📦 Subscription Plan", sorted(data["Subscription Plan"].dropna().unique()))
        unit_price = st.number_input("💰 Unit Price (₦)", min_value=0.0, step=0.1)

    num_times_purchased = st.number_input("🛒 Number of Times Purchased", min_value=0)
    total_revenue = st.number_input("💳 Total Amount Spent (₦)", min_value=0.0, step=0.1)
    data_usage = st.number_input("📶 Data Usage (MB)", min_value=0.0, step=0.1)

    submitted = st.form_submit_button("🔍 Predict")

# =============================
# 🧠 Prediction Logic
# =============================
if submitted:
    input_df = pd.DataFrame({
        'date of purchase': [str(date_of_purchase)],
        'age': [age],
        'state': [state],
        'mtn device': [mtn_device],
        'gender': [gender],
        'satisfaction rate': [satisfaction_rate],
        'customer review': [customer_review],
        'customer tenure in months': [customer_tenure],
        'subscription plan': [subscription_plan],
        'unit price': [unit_price],
        'number of times purchased': [num_times_purchased],
        'total revenue': [total_revenue],
        'data usage': [data_usage]
    })

    try:
        prediction = model.predict(input_df)[0]
        probas = model.predict_proba(input_df)[0]
        likelihood = round(float(max(probas)) * 100, 2)

        if prediction == 1:
            st.error("📛 The customer is unlikely to renew their subscription.")
            st.markdown(f"<i>This prediction is approximately <b>{likelihood:.2f}%</b> likely to be true.</i>", unsafe_allow_html=True)
            st.markdown("<i>🔁 Suggest personalized retention offers like bonus data or loyalty rewards.</i>", unsafe_allow_html=True)
        else:
            st.success("✅ The customer is likely to renew their subscription.")
            st.markdown(f"<i>This prediction is approximately <b>{likelihood:.2f}%</b> likely to be true.</i>", unsafe_allow_html=True)
            st.markdown("<i>👏 Maintain this relationship with quality service and perks.</i>", unsafe_allow_html=True)

    except ValueError as ve:
        st.error(f"Prediction failed: {ve}")

# =============================
# 🗒️ Footer
# =============================
st.markdown("---")
st.markdown("""
    <div style="font-size: 13px; color: #cccccc; text-align: center;">
        Built with ❤️ using <a href="https://streamlit.io" target="_blank" style="color: #ffcc00;">Streamlit</a><br>
        Created by <strong>Tolulope Emuleomo</strong> aka <strong>Data Professor</strong> 🧠<br>
        🔗 <a href="https://twitter.com/dataprofessor_" target="_blank" style="color: #1DA1F2;">@dataprofessor_</a> |
        <a href="https://github.com/dataprofessor290" target="_blank" style="color: #c9a0ff;">GitHub</a> |
        <a href="https://www.linkedin.com/in/tolulope-emuleomo-77a231270/" target="_blank" style="color: #0A66C2;">LinkedIn</a><br>
        💼 <span style="color: #999;">Data Scientist</span>
    </div>
""", unsafe_allow_html=True)
