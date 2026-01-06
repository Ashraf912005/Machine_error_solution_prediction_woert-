import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Machine Status Prediction",
    page_icon="⚙️",
    layout="wide"
)

# ---------------------------------
# CUSTOM CSS
# ---------------------------------
st.markdown("""
<style>
.big-title { font-size: 36px; font-weight: 700; }
.sub-title { color: #aaaaaa; font-size: 16px; }
.kpi-box {
    background-color: #1f2937;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.kpi-value {
    font-size: 26px;
    font-weight: bold;
    color: #22c55e;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# HEADER
# ---------------------------------
st.markdown("<div class='big-title'>⚙️ Machine Status / Error Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Random Forest based real-time machine status prediction</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------
# LOAD DATA
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Machine_status_code_History_New.csv", encoding="latin1")
    return df.dropna()

df = load_data()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ---------------------------------
# ENCODING
# ---------------------------------
encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

le_target = LabelEncoder()
y = le_target.fit_transform(y)

# ---------------------------------
# TRAIN–TEST SPLIT
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------
# MODEL
# ---------------------------------
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------
# MODEL ACCURACY
# ---------------------------------
y_pred = model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred) * 100

# ---------------------------------
# SIDEBAR INPUT
# ---------------------------------
st.sidebar.header("🔧 Input Parameters")
user_input = {}

for col in df.columns[:-1]:
    if col in encoders:
        option = st.sidebar.selectbox(col, encoders[col].classes_)
        user_input[col] = encoders[col].transform([option])[0]
    else:
        user_input[col] = st.sidebar.number_input(
            col,
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )

input_df = pd.DataFrame([user_input])

# ---------------------------------
# PREDICTION
# ---------------------------------
if st.sidebar.button("🔍 Predict Machine Status"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df).max()
    predicted_label = le_target.inverse_transform(prediction)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"<div class='kpi-box'><div>Predicted Status</div><div class='kpi-value'>{predicted_label}</div></div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"<div class='kpi-box'><div>Prediction Confidence</div><div class='kpi-value'>{probability*100:.2f}%</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("📋 Input Summary")

    display_df = input_df.copy()
    for col in encoders:
        display_df[col] = encoders[col].inverse_transform(display_df[col])

    st.dataframe(display_df, use_container_width=True)

# ---------------------------------
# MODEL INFO
# ---------------------------------
st.markdown("---")
st.subheader("📊 Model Information")
st.write("• Algorithm: **Random Forest Classifier**")
st.write(f"• Model Accuracy: **{model_accuracy:.2f}%**")
st.write("• Target Classes:")
st.code(dict(enumerate(le_target.classes_)))

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Python, Scikit-learn & Streamlit")
