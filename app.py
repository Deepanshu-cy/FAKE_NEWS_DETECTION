import math
import os
import pickle
import re
import string

from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"


def clean_news(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = " ".join(text.split())

    return text


def load_model_artifacts():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        raise FileNotFoundError(
            "Missing trained model artifacts. Run train_model.py first."
        )

    with open(VECTORIZER_FILE, "rb") as f:
        vectorizer = pickle.load(f)

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_model_artifacts()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/predict", methods=["POST"])
def predict():
    news = request.form.get("news", "")

    if news.strip() == "":
        return render_template(
            "index.html",
            prediction="Please enter some news."
        )

    cleaned_text = clean_news(news)
    transformed = vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed)[0]

    label = "Real News ✅" if int(prediction) == 1 else "Fake News ❌"
    confidence_score = None

    if hasattr(model, "decision_function"):
        score = model.decision_function(transformed)[0]
        confidence_score = round(1 / (1 + math.exp(-abs(score))) * 100, 2)

    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence_score,
        news=news
    )


if __name__ == "__main__":
    app.run(debug=True)
