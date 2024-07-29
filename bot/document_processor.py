import fitz  # PyMuPDF
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

import json
from PyPDF2 import PdfReader

def initialize_document_store(pdf_path):
    document_store = []
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text = page.extract_text()
            # Chunk the text if necessary
            chunks = text.split('\n\n')  # Example: Splitting by double newlines
            for chunk in chunks:
                if chunk.strip():
                    document_store.append({'text': chunk.strip()})
    
    # Save the chunks to a JSON file for later use
    with open('data/extracted_chunks.json', 'w') as f:
        json.dump(document_store, f)
    
    return document_store


import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Global variables for document vectors and vectorizer
doc_vectors = None
vectorizer = TfidfVectorizer(stop_words='english')

def retrieve_relevant_documents(query, document_store, top_n=1):
    global doc_vectors, vectorizer
    
    # Load and vectorize documents if doc_vectors is not already initialized
    if doc_vectors is None:
        with open('data/extracted_chunks.json') as f:
            document_store = json.load(f)
        document_texts = [doc['text'] for doc in document_store]
        doc_vectors = vectorizer.fit_transform(document_texts)
    
    # Transform the query to the same vector space
    query_vector = vectorizer.transform([query])
    
    # Compute cosine similarities between query and document vectors
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    
    # Get the indices of the top_n most similar documents
    relevant_doc_indices = similarities.argsort()[-top_n:][::-1]
    
    # Retrieve the relevant documents
    relevant_docs = [document_store[i]['text'] for i in relevant_doc_indices]
    return relevant_docs

