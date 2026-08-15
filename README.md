# FAKE_NEWS_DETECTION
# 📰 Fake News Detection using Machine Learning

## 📌 Project Overview

Fake News Detection is a web application developed using Python, Flask, and Machine Learning. The system predicts whether a news article is **Real** or **Fake** based on the text entered by the user.

The model is trained using the **Fake.csv** and **True.csv** datasets with **TF-IDF Vectorization** and the **Passive Aggressive Classifier**.

---

## 🚀 Features

- Detects Fake and Real news
- Simple and user-friendly interface
- Confidence score
- Responsive website
- About Page
- Contact Page
- Machine Learning based prediction

---

## 🛠 Technologies Used

- Python
- Flask
- HTML
- CSS
- JavaScript
- Scikit-Learn
- Pandas
- NumPy
- TF-IDF Vectorizer
- Passive Aggressive Classifier

---

## 📂 Project Structure

```
Fake-News-Detection/

│── app.py
│── train_model.py
│── model.pkl
│── vectorizer.pkl
│── requirements.txt
│
├── dataset/
│     ├── Fake.csv
│     └── True.csv
│
├── static/
│     ├── style.css
│     └── script.js
│
├── templates/
│     ├── index.html
│     ├── about.html
│     └── contact.html
│
└── README.md
```

---

## ⚙ Installation

### Step 1

Clone the repository

```bash
git clone https://github.com/yourusername/Fake-News-Detection.git
```

### Step 2

Go inside the project folder

```bash
cd Fake-News-Detection
```

### Step 3

Install required libraries

```bash
pip install -r requirements.txt
```

### Step 4

Train the model

```bash
python train_model.py
```

This will generate

- model.pkl
- vectorizer.pkl

---

### Step 5

Run the Flask application

```bash
python app.py
```

---

## Open in Browser

```
http://127.0.0.1:5000/
```

---

## Dataset

The project uses two datasets:

- Fake.csv
- True.csv

These datasets are placed inside the **dataset** folder.

---

## Machine Learning Workflow

```
Dataset
      ↓
Data Cleaning
      ↓
TF-IDF Vectorizer
      ↓
Passive Aggressive Classifier
      ↓
Model Training
      ↓
Prediction
```

---

## Future Improvements

- Deep Learning Model (LSTM/BERT)
- Multi-language Support
- News URL Detection
- Voice Input
- User Login System
- Prediction History
- Dark Mode
- Live News API Integration

---

## Developed By

**Deepanshu**

B.Tech Project

Generative AI

---

## License

This project is developed for educational purposes only.
