# Step 1: Resume & JD Ingestion
import PyPDF2
import spacy
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ---------------- PDF Extraction ----------------
def extract_text_from_pdf(pdf_path_or_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_path_or_file)
    for page in reader.pages:
        text += page.extract_text() + " "
    return text

# ---------------- Preprocessing ----------------
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_space]
    return " ".join(tokens)

# ---------------- Ranking Logic ----------------
def rank_candidates(resumes, jd_text):
    jd_clean = preprocess_text(jd_text)
    scores = []
    vectorizer = TfidfVectorizer()
    for i, resume in enumerate(resumes):
        resume_clean = preprocess_text(resume)
        docs = [resume_clean, jd_clean]
        tfidf = vectorizer.fit_transform(docs)
        score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
        scores.append((i+1, score))
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked

# ---------------- Streamlit UI ----------------
def main():
    st.title("AI-Powered Resume Screening System")

    jd_input = st.text_area("Paste Job Description here")

    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

    if uploaded_files and jd_input:
        resumes = []
        for file in uploaded_files:
            resumes.append(extract_text_from_pdf(file))
        ranking = rank_candidates(resumes, jd_input)
        st.subheader("Candidate Ranking")
        for candidate, score in ranking:
            st.write(f"Candidate {candidate}: {round(score, 3)}")

if __name__ == "__main__":
    main()
