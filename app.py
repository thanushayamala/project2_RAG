import streamlit as st
import os
import faiss
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# --- RAG UTILITY FUNCTIONS ---
# This section contains the core logic for the chatbot.

# Initialize a cache for the RAG components to avoid reloading on every interaction.
@st.cache_resource
def load_rag_components():
    """
    Loads and caches the Sentence Transformer, FAISS index, and chunks.
    This function will only run once for efficiency.
    """
    try:
        # Check if the vectorstore exists
        if not os.path.exists("vectorstore/faiss_index.bin"):
            st.error("Vectorstore not found! Please run 'python run_once.py' first.")
            return None, None, None

        # Load the sentence transformer model
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load the FAISS index and the text chunks
        index = faiss.read_index("vectorstore/faiss_index.bin")
        chunks = np.load("vectorstore/chunks.npy", allow_pickle=True)
        
        return embed_model, index, chunks
    except Exception as e:
        st.error(f"Error loading RAG components: {e}")
        return None, None, None

def retrieve_and_answer(query, embed_model, index, chunks, top_k=2):
    """
    Performs the RAG pipeline:
    1. Embeds the user's query.
    2. Searches the FAISS index for relevant text chunks.
    3. Generates a response using a language model with the retrieved context.
    
    Args:
        query (str): The user's question.
        embed_model: The Sentence Transformer model.
        index: The FAISS index.
        chunks (list): The list of text chunks.
        top_k (int): The number of top relevant chunks to retrieve.
        
    Returns:
        str: The generated answer.
    """
    # Embed the user's query
    query_embedding = embed_model.encode([query])
    
    # Search the FAISS index for the most similar chunks
    distances, indices = index.search(np.array(query_embedding), k=top_k)
    
    # Get the relevant chunks based on the indices
    context = " ".join([chunks[i] for i in indices[0]])
    
    # Initialize a text generation pipeline with the Mistral model
    # Note: This is a placeholder model from the reference code. You can change this to another suitable model.
    generator = pipeline("text2text-generation", model="google/flan-t5-small")
    
    # Create the prompt for the language model, including the retrieved context
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    
    # Generate the answer using the language model
    response = generator(prompt, max_length=100, do_sample=False)
    
    return response[0]['generated_text']

# --- STREAMLIT APP LAYOUT AND LOGIC ---
st.set_page_config(page_title="PDF Chatbot", layout="centered")

# Load components and check for errors
embed_model, index, chunks = load_rag_components()

if embed_model is None or index is None or chunks is None:
    st.stop()  # Stop the app if components failed to load

st.title("📚 RAG-Based PDF Chatbot")
st.markdown("Ask me questions based on the PDFs you uploaded to the `docs` folder!")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is a Large Language Model?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate the assistant's response
    with st.spinner("Thinking..."):
        try:
            response = retrieve_and_answer(prompt, embed_model, index, chunks)
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
                
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
