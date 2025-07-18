import joblib

# Load model and vectorizer
model = joblib.load("logreg_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Input
print("🔎 Paste your news article (multi-line supported). Press ENTER twice to submit:")
lines = []
while True:
    line = input()
    if line == "":
        break
    lines.append(line)

news = " ".join(lines)

# Transform & Predict
vector = tfidf.transform([news])
prob = model.predict_proba(vector)[0]
label = "REAL" if prob[1] >= 0.5 else "FAKE"

# Output
print(f"\n🧠 Prediction: {label}")
print(f"✅ Confidence - Real: {prob[1]*100:.2f}%, Fake: {prob[0]*100:.2f}%")
