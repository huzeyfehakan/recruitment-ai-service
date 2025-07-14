from fastapi import FastAPI, UploadFile, File
from cv_parser import parse_pdf
from skill_classifier import classify_skills
import shutil
import uuid
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Recruitment AI System is running!"}

@app.post("/classify-cv")
async def classify_cv(file: UploadFile = File(...)):
    # PDF dosyasını geçici bir yere kaydet
    temp_filename = f"{uuid.uuid4()}.pdf"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # CV metnini parse et
    text = parse_pdf(temp_filename)

    # Geçici dosyayı sil
    os.remove(temp_filename)

    # Ollama ile becerileri sınıflandır
    result = classify_skills(text)
    return {"skills": result}