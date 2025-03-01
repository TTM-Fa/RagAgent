import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from mistralai import Mistral, UserMessage
import pickle
import os

# -----------------  Loading API Key -----------------
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]  # API Key from Streamlit Secrets

# ----------------- Defining URLs -----------------
policies = {
    "Transfer Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy",
    "Student Appeals Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
    "Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
    "Registration Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
    "Sport and Wellness Facilities": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and",
    "Scholarship and Financial Assistance": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance",
    "Credit Hour Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
    "Academic Annual Leave Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy",
    "Examination Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/examination-policy",
    "Library Study Room Booking": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/library-study-room-booking-procedure",
}

TEXT_FILE = "combined_policies.txt"

def load_and_chunk_text():
    """Read saved combined text file and split into chunks."""
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    chunk_size = 512
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

all_chunks = load_and_chunk_text()

# Load FAISS index from .index file
index = faiss.read_index("faiss_index.index")

def get_text_embedding(list_txt_chunks):    
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])    
    embeddings_batch_response = client.embeddings.create(model="mistral-embed",       
                                                        inputs=list_txt_chunks) 
    return embeddings_batch_response.data

def mistral(user_message, model="mistral-small-latest", is_json=False):
    model = "mistral-large-latest"
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    messages = [
        UserMessage(content=user_message),
    ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
    )
    return chat_response.choices[0].message.content


# ----------------- Streamlit UI -----------------
st.title("UDST Policy Query System")

col1, col2 = st.columns([1, 2])

# Left Panel: List of Policies
with col1:
    st.header("Policies")
    for name, url in policies.items():
        st.markdown(f"- [{name}]({url})")

# Right Panel: Query Processing
with col2:
    st.header("Query")
    user_query = st.text_input("Enter your question:")

    if user_query:
        query_embedding = np.array([get_text_embedding([user_query])[0].embedding])

        D, I = index.search(query_embedding, k=3)
        retrieved_chunks = [all_chunks[i] for i in I[0]]

        retrieved_chunk = [all_chunks[i] for i in I.tolist()[0]] 
        prompt = f"""
        Context information is below.
        ---------------------
        {retrieved_chunk}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {user_query}
        Answer:
        """

        response = mistral(prompt)
        answer = response

        # Display result
        st.header("Result of Query")
        st.write(answer)
