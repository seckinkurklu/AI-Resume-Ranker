import fitz  # PyMuPDF
import re

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model)

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


def extract_keywords(text, top_n=10):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n

    )
    return [kw[0] for kw in keywords]

def compare_keywords(resume_text, job_keywords):
    present = [kw for kw in job_keywords if kw.lower() in resume_text.lower()]
    missing = [kw for kw in job_keywords if kw.lower() not in resume_text.lower()]
    return present, missing

def get_section_scores(resume_text, job_desc_text):
    sections = {
        "Skills": extract_section(job_desc_text, ["skills", "requirements"]),
        "Experience": extract_section(job_desc_text, ["experience", "responsibilities"]),
        "Education": extract_section(job_desc_text, ["education", "skills"]),
    }

    scores = {}
    for name, section in sections.items():
        if section.strip():
            score = get_similarity_score(resume_text, section)
            scores[name] = score
        else:
            scores[name] = None

        return scores

def extract_section(text, keywords):
    text = text.lower()
    for keyword in keywords:
        if keyword in text:
            start = text.find(keyword)
            return text[start:start + 500]
    return ""