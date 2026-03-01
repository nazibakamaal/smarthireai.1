"""
SmartHireAI - AI Resume Screening System
Production-ready Flask application for resume-job description matching
"""
import os
import re
from io import BytesIO
from werkzeug.utils import secure_filename

import pdfplumber
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")

ALLOWED_EXTENSIONS = {"pdf"}


def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text: str) -> str:
    """
    Basic NLP preprocessing: lowercase, remove punctuation, normalize whitespace.
    """
    if not text or not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r"[^\w\s]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    text_parts = []
    try:
        with pdfplumber.open(BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        raise ValueError(f"Failed to extract PDF text: {e}") from e
    return "\n".join(text_parts) if text_parts else ""


def calculate_match_score(resume_text: str, job_description: str) -> float:
    """
    Compare resume with job description using TF-IDF and cosine similarity.
    Returns score as percentage (0-100).
    """
    if not resume_text.strip() or not job_description.strip():
        return 0.0

    cleaned_resume = clean_text(resume_text)
    cleaned_job = clean_text(job_description)

    if not cleaned_resume or not cleaned_job:
        return 0.0

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_job])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Convert to percentage (0-1 range to 0-100)
    return round(min(max(similarity * 100, 0), 100), 2)


def rank_candidates(resumes_data: list[dict], job_description: str) -> list[dict]:
    """
    Score and rank multiple resumes against job description.
    Returns sorted list with rank, score, and filename.
    """
    results = []
    for item in resumes_data:
        score = calculate_match_score(item["text"], job_description)
        results.append({
            "filename": item["filename"],
            "score": score,
            "text_preview": item["text"][:200] + "..." if len(item["text"]) > 200 else item["text"],
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # Add rank
    for i, r in enumerate(results, start=1):
        r["rank"] = i

    return results


@app.route("/")
def index():
    """Render upload form."""
    return render_template("index.html")


@app.route("/screen", methods=["POST"])
def screen():
    """Process uploaded resumes and job description."""
    job_description = request.form.get("job_description", "").strip()
    files = request.files.getlist("resumes")

    if not job_description:
        flash("Please enter a job description.", "error")
        return redirect(url_for("index"))

    if not files or all(f.filename == "" for f in files):
        flash("Please upload at least one PDF resume.", "error")
        return redirect(url_for("index"))

    resumes_data = []
    errors = []

    for file in files:
        if file and file.filename and allowed_file(file.filename):
            try:
                content = file.read()
                text = extract_text_from_pdf(content)
                if not text.strip():
                    errors.append(f"{file.filename}: No text could be extracted.")
                else:
                    resumes_data.append({"filename": secure_filename(file.filename), "text": text})
            except ValueError as e:
                errors.append(f"{file.filename}: {e}")
            except Exception as e:
                errors.append(f"{file.filename}: Unexpected error - {e}")
        elif file and file.filename:
            errors.append(f"{file.filename}: Only PDF files are allowed.")

    if errors:
        for err in errors:
            flash(err, "error")

    if not resumes_data:
        flash("No valid resumes could be processed.", "error")
        return redirect(url_for("index"))

    ranked = rank_candidates(resumes_data, job_description)
    top_candidates = [r for r in ranked if r["rank"] <= 3]  # Top 3

    return render_template(
        "result.html",
        results=ranked,
        top_candidates=top_candidates,
        job_description_preview=job_description[:300] + "..." if len(job_description) > 300 else job_description,
    )


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
