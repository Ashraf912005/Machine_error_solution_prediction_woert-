import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Machine Status Prediction",
    layout="centered"
)

st.title("⚙️ Machine Status / Error Prediction App")
st.write("Predict machine status using Random Forest model")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Machine_status_code_History_New.csv", encoding="latin1")
    df = df.dropna()
    return df

df = load_data()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# -----------------------------
# ENCODING
# -----------------------------
encoders = {}

for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

# Target encoding
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X, y)

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("🔧 Input Parameters")

user_input = {}

for col in df.columns[:-1]:
    if col in encoders:
        options = encoders[col].classes_
        selected = st.sidebar.selectbox(f"{col}", options)
        user_input[col] = encoders[col].transform([selected])[0]
    else:
        value = st.sidebar.number_input(
            f"{col}",
            min_value=float(df[col].min()),
            max_value=float(df[col].max()),
            value=float(df[col].mean())
        )
        user_input[col] = value

input_df = pd.DataFrame([user_input])

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict Machine Status"):
    prediction = model.predict(input_df)
    prediction_label = le_target.inverse_transform(prediction)

    st.success(f"✅ Predicted Output: **{prediction_label[0]}**")

# -----------------------------
# MODEL INFO
# -----------------------------
st.markdown("---")
st.subheader("📊 Model Information")

st.write("• Algorithm: Random Forest Classifier")
st.write("• Number of Trees:", model.n_estimators)
st.write("• Target Classes:")

st.write(list(le_target.classes_))

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Python, Scikit-learn & Streamlit")
