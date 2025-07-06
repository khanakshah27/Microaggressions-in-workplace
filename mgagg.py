import os
import pandas as pd
import spacy
import joblib
import numpy as np
import faiss
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer

os.system("python -m spacy download en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

class SpacyPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return [" ".join([token.lemma_ for token in nlp(text.lower()) if not token.is_punct]) for text in X]

def train_and_save_pipeline():
    print("Loading and preprocessing data...")
    df = pd.read_csv("micro_agg.csv", encoding='ISO-8859-1')
    df['ptext'] = df['speech'].apply(lambda x: " ".join(
        [token.lemma_ for token in nlp(x.lower()) if not token.is_stop and not token.is_punct]
    ))

    X = df['speech']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("⚙️ Building and training pipeline...")
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

    print("Building FAISS index for RAG explanations...")
    kb_df = pd.read_csv("rag_kb.csv", encoding='ISO-8859-1')
    kb = kb_df['explanation'].dropna().tolist()
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    kb_embeddings = embedder.encode(kb)
    index = faiss.IndexFlatL2(kb_embeddings.shape[1])
    index.add(kb_embeddings)
    print("FAISS index built in memory")

def load_pipeline():
    return joblib.load("microagg_model.pkl")

def load_kb_embedder_index():
    kb_df = pd.read_csv("rag_kb.csv", encoding="ISO-8859-1")
    kb = kb_df['explanation'].dropna().tolist()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    kb_embeddings = embedder.encode(kb)
    index = faiss.IndexFlatL2(kb_embeddings.shape[1])
    index.add(kb_embeddings)
    return kb, embedder, index

def classify_and_explain(text, model, kb, embedder, index):
    prediction = model.predict([text])[0]
    if prediction == 1:
        query_vec = embedder.encode([text])
        _, I = index.search(np.array(query_vec), k=1)
        explanation = kb[I[0][0]]
        return "Microaggression", explanation
    else:
        return "Not a microaggression", None

