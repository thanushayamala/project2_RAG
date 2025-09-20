import os
import faiss
import numpy as np
import PyPDF2
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Define paths for the docs and vectorstore folders
DOCS_FOLDER = "docs"
VECTORSTORE_FOLDER = "vectorstore"

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF file.
    
    Args:
        pdf_path (str): The path to the PDF file.
        
    Returns:
        str: The concatenated text from all pages of the PDF.
    """
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=300):
    """
    Splits a long string of text into smaller chunks.
    
    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The desired size of each chunk.
        
    Returns:
        list: A list of text chunks.
    """
    # Simple chunking by splitting words and joining them until the chunk size is reached
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= chunk_size:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def process_and_build_vectorstore():
    """
    Main function to process PDFs, create embeddings, and build the FAISS vectorstore.
    """
    # Create the vectorstore directory if it doesn't exist
    if not os.path.exists(VECTORSTORE_FOLDER):
        os.makedirs(VECTORSTORE_FOLDER)

    all_chunks = []
    print("Reading and chunking PDFs from the 'docs' folder...")
    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(DOCS_FOLDER, filename)
            document_text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(document_text)
            all_chunks.extend(chunks)
            print(f"Processed '{filename}'. Total chunks: {len(chunks)}")
    
    if not all_chunks:
        print("No PDF files found in the 'docs' folder. Please add your documents and try again.")
        return

    print(f"\nTotal chunks from all documents: {len(all_chunks)}")
    
    # Initialize the sentence transformer model for embeddings
    print("Initializing Sentence Transformer model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Create embeddings for all chunks
    print("Creating embeddings for all text chunks...")
    embeddings = embed_model.encode(all_chunks)
    
    # Create a FAISS index for efficient searching
    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # Using L2 distance for similarity search
    index.add(np.array(embeddings))
    
    # Save the FAISS index and the chunks
    faiss.write_index(index, os.path.join(VECTORSTORE_FOLDER, "faiss_index.bin"))
    np.save(os.path.join(VECTORSTORE_FOLDER, "chunks.npy"), np.array(all_chunks))
    
    print("\nVectorstore built successfully!")
    print("FAISS index saved to 'vectorstore/faiss_index.bin'")
    print("Text chunks saved to 'vectorstore/chunks.npy'")

if __name__ == "__main__":
    process_and_build_vectorstore()
