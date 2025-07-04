import pandas as pd
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import streamlit as st

# Load spacy model at global scope
try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class SpacyPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return [
            " ".join([token.lemma_ for token in nlp(text.lower()) if not token.is_punct])
            for text in X
        ]

@st.cache_resource(show_spinner="Training model...")
def train_pipeline():
    df = pd.read_csv("micro_agg.csv", encoding='ISO-8859-1')
    X = df['speech']
    y = df['label']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('spacy', SpacyPreprocessor()),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)),
        ('sgd', SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=1e-5,
                              max_iter=1000, class_weight='balanced', random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

def classify(text, model):
    prediction = model.predict([text])[0]
    return "Microaggression" if prediction == 1 else "Not a microaggression"

# Streamlit UI
st.title("🚨 Microaggressions Detector")
st.write("Enter workplace text to check for microaggressions.")

model = train_pipeline()
user_input = st.text_area("✍️ Enter your sentence:")

if st.button("Classify"):
    if user_input.strip():
        result = classify(user_input, model)
        st.success(f"**Result:** {result}")
    else:
        st.warning("Please enter some text.")
