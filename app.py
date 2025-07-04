import os
import string
import pandas as pd
import numpy as np
import streamlit as st

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r"\w+")
        self.punct_set = set(string.punctuation)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [
            " ".join(
                self.lemmatizer.lemmatize(word.lower())
                for word in self.tokenizer.tokenize(text)
                if word not in self.punct_set
            )
            for text in X
        ]

@st.cache_resource(show_spinner="Training model...")
def train_pipeline():
    df = pd.read_csv("micro_agg.csv", encoding='ISO-8859-1')
    X = df['speech']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('preprocess', NLTKPreprocessor()),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)),
        ('sgd', SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=1e-5,
                              max_iter=1000, class_weight='balanced', random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    st.write("###Model Evaluation")
    st.text(classification_report(y_test, y_pred))
    st.write(f"**Accuracy:** `{accuracy_score(y_test, y_pred):.4f}`")

    return pipeline

def classify(text, model):
    prediction = model.predict([text])[0]
    return "Microaggression" if prediction == 1 else "Not a microaggression"

st.title("Microaggressions Detector")
st.write("Enter text below to check whether it contains a microaggression.")

model = train_pipeline()
user_input = st.text_area("Enter your sentence:")

if st.button("Classify"):
    if user_input.strip():
        result = classify(user_input, model)
        st.success(f"**Result:** {result}")
    else:
        st.error("Please enter a valid sentence.")

