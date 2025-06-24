import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # remove excess whitespace
    text = re.sub(r'[^a-zA-Z0-9., ]', '', text)  # basic cleaning
    return text.strip().lower()

def get_similarity_score(resume_text, job_desc_text):
    embeddings = model.encode([resume_text, job_desc_text])
    sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(sim_score), 4)
