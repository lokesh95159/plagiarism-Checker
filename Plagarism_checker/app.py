import os
import re
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import PyPDF2
import docx

# Function to preprocess text (tokenization, stopword removal, etc.)
def preprocess_text(text):
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Function to calculate cosine similarity between two documents
def calculate_cosine_similarity(doc1, doc2):
    # Convert documents into tf-idf vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_matrix[0][0]

# Function to check plagiarism between two documents
def check_plagiarism(doc1, doc2):
    # Preprocess the text (tokenization, cleaning, etc.)
    doc1 = preprocess_text(doc1)
    doc2 = preprocess_text(doc2)

    # Calculate similarity
    similarity = calculate_cosine_similarity(doc1, doc2)
    return similarity

# Function to read text from PDF files
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to read text from Word files
def read_word(file):
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Streamlit user interface for uploading and checking documents
def plagiarism_check_interface():
    st.title("Plagiarism Detection System")

    # File upload for two documents
    doc1 = st.file_uploader("Upload the first document", type=["txt", "pdf", "docx"])
    doc2 = st.file_uploader("Upload the second document", type=["txt", "pdf", "docx"])

    if doc1 is not None and doc2 is not None:
        # Read the contents of the uploaded files
        if doc1.type == "application/pdf":
            doc1_text = read_pdf(doc1)
        elif doc1.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc1_text = read_word(doc1)
        else:
            doc1_text = doc1.read().decode("utf-8")

        if doc2.type == "application/pdf":
            doc2_text = read_pdf(doc2)
        elif doc2.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc2_text = read_word(doc2)
        else:
            doc2_text = doc2.read().decode("utf-8")

        # Check for plagiarism
        similarity = check_plagiarism(doc1_text, doc2_text)

        # Display the results
        st.write(f"Cosine Similarity between the two documents: {similarity * 100:.2f}%")
        if similarity > 0.8:
            st.warning("High plagiarism detected!")
        elif similarity > 0.5:
            st.warning("Medium plagiarism detected!")
        else:
            st.success("Low plagiarism detected!")

if __name__ == "__main__":
    plagiarism_check_interface()