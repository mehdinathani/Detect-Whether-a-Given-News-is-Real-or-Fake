from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from main import df


# Tokenize text data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df["content"])
sequences = tokenizer.texts_to_sequences(df["content"])
X_pad = pad_sequences(sequences, maxlen=300)

# Labels
y = df["label"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=300),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n🚀 Training LSTM model...")
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
