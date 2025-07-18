from transformers import pipeline
from sklearn.metrics import classification_report
from main import df

# Load pre-trained BERT pipeline
bert_clf = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Sample prediction
sample_text = df["content"].iloc[0]
print("\n🤖 BERT prediction on sample text:")
print(bert_clf(sample_text))
