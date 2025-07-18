# streamlit_app.py
import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("logreg_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="News Real/Fake Classifier", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("Paste a news article or headline below:")

news = st.text_area("News content", height=300)

if st.button("Predict"):
    if not news.strip():
        st.warning("Please enter some news content.")
    else:
        vector = tfidf.transform([news])
        prob = model.predict_proba(vector)[0]
        label = "REAL" if prob[1] >= 0.5 else "FAKE"

        st.success(f"🧠 Prediction: **{label}**")
        st.info(f"✅ Confidence: **Real: {prob[1]*100:.2f}%**, Fake: **{prob[0]*100:.2f}%**")
