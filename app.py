import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ LOGIN SYSTEM ------------------
def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid credentials")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return pickle.load(open("model/model.pkl", "rb"))

# ------------------ MAIN APP ------------------
def main():
    st.set_page_config(page_title="Iris Predictor", layout="centered")

    st.title("🌸 Iris Flower Prediction App")
    st.markdown("### Enter flower measurements")

    model = load_model()

    # -------- INPUT UI --------
    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
        sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)

    with col2:
        petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
        petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # -------- PREDICTION --------
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        st.success(f"🌼 Predicted Species: {prediction}")

        # -------- PROBABILITY DISPLAY --------
        st.subheader("📊 Prediction Probability")

        classes = model.classes_
        prob_df = pd.DataFrame({
            "Species": classes,
            "Probability": probabilities
        })

        st.dataframe(prob_df)

        # -------- GRAPH --------
        st.subheader("📈 Probability Chart")

        fig, ax = plt.subplots()
        ax.bar(classes, probabilities)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")

        st.pyplot(fig)

# ------------------ APP FLOW ------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    main()