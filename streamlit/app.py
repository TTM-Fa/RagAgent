import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from mistralai import Mistral

# Define policies and URLs
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

# Load API Key (Replace with your actual API key)
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]

# Initialize Mistral API Client
client = Mistral(api_key=MISTRAL_API_KEY)

# Function to scrape and clean text from URLs
def scrape_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join([p.get_text() for p in paragraphs])
        return text.strip()
    return ""

# Load all policies into a dictionary
policy_texts = {name: scrape_text(url) for name, url in policies.items()}

# Chunking function
def chunk_text(text, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Prepare document embeddings
chunked_data = {name: chunk_text(text) for name, text in policy_texts.items()}

# Flatten chunks for embedding
all_chunks = []
chunk_to_policy = {}
for policy, chunks in chunked_data.items():
    for chunk in chunks:
        chunk_to_policy[len(all_chunks)] = policy
        all_chunks.append(chunk)

# Get embeddings for all chunks
def get_text_embedding(text_list):
    response = client.embeddings.create(model="mistral-embed", inputs=text_list)
    return np.array([entry.embedding for entry in response.data])

text_embeddings = get_text_embedding(all_chunks)

# Create FAISS index
d = len(text_embeddings[0])  # Dimension
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

# Streamlit UI
st.title("UDST Policy Query System")

col1, col2 = st.columns([1, 2])

# Left Panel: Policies
with col1:
    st.header("Policies")
    for name, url in policies.items():
        st.markdown(f"- [{name}]({url})")

# Right Panel: Query Section
with col2:
    st.header("Query")
    user_query = st.text_input("Enter your question:")

    if user_query:
        # Get query embedding
        query_embedding = np.array([get_text_embedding([user_query])[0]])

        # Retrieve similar chunks
        D, I = index.search(query_embedding, k=3)  # Retrieve top 3 matches
        retrieved_chunks = [all_chunks[i] for i in I[0]]
        sources = [chunk_to_policy[i] for i in I[0]]

        # Prepare context for response
        context = "\n\n".join(retrieved_chunks)

        # Generate response
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
