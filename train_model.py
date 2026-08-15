import os
import pickle
import re
import string

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "dataset", "news_dataset.csv")
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_FILE = os.path.join(BASE_DIR, "vectorizer.pkl")

# Step 1: Text cleaning (basic NLP preprocessing)
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\d+", "", text)  # remove numbers
    return " ".join(text.split())

# Step 2: Load dataset
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    # Expecting 'text' and 'label' columns in Kaggle dataset
    df["text"] = df["text"].astype(str).apply(clean_text)
    return df[["text", "label"]]

# Step 3: Train model
def train_model():
    data = load_dataset()
    X = data["text"]
    y = data["label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train classifier
    model = SGDClassifier(loss="hinge", random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save model + vectorizer
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)

    print("Model and vectorizer saved successfully.")

if __name__ == "__main__":
    train_model()
