
import streamlit as st
import joblib

from preprocessing import prepare_message

# Load trained models
bow_vectorizer = joblib.load(
    "exported_model/bow_vectorizer.pkl"
)

bow_logistic_model = joblib.load(
    "exported_model/bow_logistic_model.pkl"
)

# Streamlit page configuration
st.set_page_config(
    page_title="Telegram Spam Detection",
    page_icon="📩",
    layout="centered"
)

st.title("📩 Telegram Spam Detection")

st.write(
    "Enter a Telegram message to determine "
    "whether it is spam or legitimate."
)

# User input
message = st.text_area(
    "Enter your Telegram message:",
    placeholder="Type or paste your message here..."
)

if st.button("Classify Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        # 1. Preprocess message
        processed_message = prepare_message(message)

        # 2. Convert text into BoW features
        message_vector = bow_vectorizer.transform(
            [processed_message]
        )

        # 3. Predict classification
        prediction = bow_logistic_model.predict(
            message_vector
        )[0]

        # 4. Display result
        if prediction == "spam":
            st.error("⚠️ This message is classified as SPAM.")

        else:
            st.success("✅ This message is classified as HAM.")

        # Optional: Display processed message
        with st.expander("View processed message"):
            st.write(processed_message)
