import os
import pandas as pd
import spacy
import joblib
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

os.system("python -m spacy download en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

class SpacyPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return [" ".join([token.lemma_ for token in nlp(text.lower()) if not token.is_punct]) for text in X]

def train_and_save_pipeline():
    print("Loading and preprocessing data...")
    df = pd.read_csv("micro_agg.csv", encoding='ISO-8859-1')

    X = df['speech']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Building and training pipeline...")
    pipeline = Pipeline([
        ('spacy', SpacyPreprocessor()),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)),
        ('sgd', SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=1e-5,
                              max_iter=1000, class_weight='balanced', random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("\nEvaluation:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    joblib.dump(pipeline, "microagg_model.pkl")
    print("Model saved as 'microagg_model.pkl'")

def load_model(path="microagg_model.pkl"):
    return joblib.load(path)

def predict_text(model, text):
    prediction = model.predict([text])[0]
    return bool(prediction)

