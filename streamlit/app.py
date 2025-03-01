import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from mistralai import Mistral
import pickle
import os

# ----------------- Step 1: Load API Key Securely -----------------
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]  # API Key from Streamlit Secrets
client = Mistral(api_key=MISTRAL_API_KEY)

# ----------------- Step 2: Define URLs & Fetch Data -----------------
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

# ----------------- Step 3: Web Scraping Function -----------------
def scrape_text(url):
    """Fetch and clean text from policy web pages."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join([p.get_text() for p in paragraphs])
        return text.strip()
    return ""

# Fetch policy texts
policy_texts = {name: scrape_text(url) for name, url in policies.items()}

# ----------------- Step 4: Chunking Function -----------------
def chunk_text(text, chunk_size=512):
    """Break text into smaller chunks of specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

chunked_data = {name: chunk_text(text) for name, text in policy_texts.items()}

# Flatten chunks and track source policy
all_chunks = []
chunk_to_policy = {}
for policy, chunks in chunked_data.items():
    for chunk in chunks:
        chunk_to_policy[len(all_chunks)] = policy
        all_chunks.append(chunk)

# ----------------- Step 5: Generate or Load Embeddings -----------------
EMBEDDING_FILE = "embeddings_faiss.pkl"
FAISS_INDEX_FILE = "faiss_index.idx"

def get_text_embedding(text_list):
    """Generate embeddings using Mistral API."""
    response = client.embeddings.create(model="mistral-embed", inputs=text_list)
    return np.array([entry.embedding for entry in response.data])

def save_faiss_index(index, file_path):
    """Save FAISS index to file."""
    faiss.write_index(index, file_path)

def load_faiss_index(file_path):
    """Load FAISS index from file."""
    if os.path.exists(file_path):
        return faiss.read_index(file_path)
    return None

# Check if embeddings already exist
if os.path.exists(EMBEDDING_FILE) and os.path.exists(FAISS_INDEX_FILE):
    # Load embeddings & FAISS index
    with open(EMBEDDING_FILE, "rb") as f:
        text_embeddings = pickle.load(f)
    index = load_faiss_index(FAISS_INDEX_FILE)
else:
    # Generate embeddings from scratch
    text_embeddings = get_text_embedding(all_chunks)

    # Save embeddings for future runs
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(text_embeddings, f)

    # Create FAISS index
    d = len(text_embeddings[0])
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    # Save FAISS index
    save_faiss_index(index, FAISS_INDEX_FILE)

# ----------------- Step 6: Streamlit UI -----------------
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
        # Generate query embedding
        query_embedding = np.array([get_text_embedding([user_query])[0]])

        # Retrieve similar chunks from FAISS
        D, I = index.search(query_embedding, k=3)
        retrieved_chunks = [all_chunks[i] for i in I[0]]
        sources = [chunk_to_policy[i] for i in I[0]]

        # Prepare prompt for Mistral AI
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""
        Context information is below.
        ---------------------
        {context}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {user_query}
        Answer:
        """

        response = client.chat.complete(model="mistral-large-latest", messages=[{"role": "user", "content": prompt}])
        answer = response.choices[0].message.content

        # Display result
        st.header("Result of Query")
        st.write(answer)

        # Display sources
        st.subheader("Sources:")
        for source in set(sources):
            st.markdown(f"- **{source}**")
