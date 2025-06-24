from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from utils import extract_text_from_pdf, clean_text, get_similarity_score
import shutil
import os

app = FastAPI()

# Mount static files (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/rank-resume/")
async def rank_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    # Save uploaded resume temporarily
    temp_file_path = "temp_resume.pdf"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Extract and clean text
        resume_raw = extract_text_from_pdf(temp_file_path)
        resume_clean = clean_text(resume_raw)
        job_clean = clean_text(job_description)

        # Calculate similarity
        score = get_similarity_score(resume_clean, job_clean)
        feedback = (
            "✅ Strong match! Your resume aligns well with the job."
            if score > 0.7
            else "📝 Consider tailoring your resume more closely to the job description."
        )

        return {
            "similarity_score": score,
            "feedback": feedback
        }

    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
