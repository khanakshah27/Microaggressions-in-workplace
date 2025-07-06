

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_pipeline():
    return joblib.load("microagg_model.pkl")

@st.cache_resource
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

model = load_pipeline()
kb, embedder, index = load_kb_embedder_index()


st.title("Microaggression Detector")
st.write("Enter a sentence and we'll tell you if it may contain a microaggression.")

user_input = st.text_input("Enter your sentence:")

if user_input:
    result, explanation = classify_and_explain(user_input, model, kb, embedder, index)
    st.subheader("Prediction:")
    st.success(result)
    if explanation:
        st.subheader("Explanation:")
        st.info(explanation)
