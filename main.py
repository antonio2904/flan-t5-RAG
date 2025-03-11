from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import streamlit as st
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Sample document corpus (replace with your own knowledge base)
documents = [
    "Paris is the capital of France. It is known for the Eiffel Tower.",
    "Berlin is the capital of Germany. It has a famous wall.",
    "Madrid is the capital of Spain. It is famous for its art museums.",
    "Rome is the capital of Italy. It is home to the Colosseum.",
    "Solar system has 9 planets.",
    "A year has 365 days."
]

# Tokenize the documents
tokenized_corpus = [word_tokenize(doc.lower()) for doc in documents]

# Initialize BM25 Retriever
bm25 = BM25Okapi(tokenized_corpus)

def retrieve(query, top_n=1):
    """Retrieve top-n relevant documents using BM25."""
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_docs = sorted(zip(scores, documents), reverse=True)[:top_n]
    return [doc for _, doc in top_docs]

# Load FLAN-T5 Base model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu")  # Force CPU execution

def generate_answer(query):
    """Retrieve relevant context and generate an answer."""
    retrieved_docs = retrieve(query, top_n=1)
    context = retrieved_docs[0] if retrieved_docs else "No relevant information found."

    # Format input for FLAN-T5
    input_text = f"Context: {context} \nQuestion: {query}"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).input_ids

    # Generate answer
    outputs = outputs = model.generate(
    	input_ids,
    	max_length=20,  # Increase output length
    	min_length=5,   # Ensure a reasonable response length
    	temperature=0.5,  # Introduce variation in responses (lower = more deterministic)
    	top_p=0.9,       # Use nucleus sampling for diversity
    	num_return_sequences=1,  # Return a single best response
    	repetition_penalty=1.5,  # Reduce repetition
    	do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.header("Flan T5")
# Input text
input_text = st.text_input("Query:")

if st.button("Generate response", type="primary"):
    # Generate output
    st.text_area("Response:", generate_answer(input_text))
