import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression


# Load data
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add label
fake_df['label'] = 0
true_df['label'] = 1

# Combine and shuffle
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

print("✅ Loaded both datasets:")
print(f"Fake news shape: {fake_df.shape}")
print(f"Real news shape: {true_df.shape}")

# Step 2: Add labels
fake_df["label"] = 0  # Fake news
true_df["label"] = 1  # Real news

print("\n📌 Sample from fake_df with label:")
print(fake_df[["title", "label"]].head(2))

print("\n📌 Sample from true_df with label:")
print(true_df[["title", "label"]].head(2))

# Step 3: Combine both datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle for randomness
df = df.sample(frac=1).reset_index(drop=True)

print("\n✅ Combined dataset shape:", df.shape)

# Step 4: Display first five rows
print("\n📊 First 5 rows:")
print(df.head())


# Step 5: Show data types
print("\n🔍 Column Data Types:")
print(df.dtypes)


# Step 6: Show selected columns
print("\n🎯 Key Features Preview (title, text, label):")
print(df[["title", "text", "label"]].head())


# Step 1: Drop rows with missing values (if any)
initial_shape = df.shape
df.dropna(inplace=True)
final_shape = df.shape

print(f"\n🧹 Dropped missing values (if any). Shape before: {initial_shape}, after: {final_shape}")


# Step 2: Merge 'title' and 'text' into a new column 'content'
df["content"] = df["title"] + " " + df["text"]

print("\n🧠 Merged content preview:")
print(df[["title", "text", "content"]].head(2))


# Step 1: Split content and labels
X = df["content"]  # Features
y = df["label"]    # Labels

print(f"\n🎯 Features shape: {X.shape}")
print(f"🟡 Labels distribution:\n{y.value_counts()}")


from sklearn.feature_extraction.text import TfidfVectorizer

# Step 2: Convert text to TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

X_tfidf = tfidf.fit_transform(X)

print(f"\n📐 TF-IDF feature matrix shape: {X_tfidf.shape}")


from sklearn.model_selection import train_test_split

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42)

print("\n✅ Dataset split summary:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")


from sklearn.linear_model import LogisticRegression

# Step 1: Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\n✅ Logistic Regression model trained.")


# Step 2: Predict test set results
y_pred = model.predict(X_test)

print("\n📢 Predictions on test set generated.")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Step 3: Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Step 4: Print all metrics
print("\n📈 Model Performance Metrics:")
print(f"✅ Accuracy:  {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall:    {recall:.4f}")
print(f"✅ F1 Score:  {f1:.4f}")
print(f"✅ ROC-AUC:   {roc_auc:.4f}")


import sklearn.linear_model

# Save model and vectorizer for reuse in test.py
joblib.dump(model, "logreg_model.pkl", compress=3)
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved for testing.")

