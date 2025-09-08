import os
import httpx
import fitz
import json
from .exceptions import InvalidRequestError, UnprocessableContentError

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

EMBEDDING_MODEL = "nomic-embed-text"
EXTRACTION_MODEL = "phi4-mini-reasoning:3.8b"


async def download_and_validate_cv(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
        except httpx.RequestError as e:
            raise InvalidRequestError(f"given url is not valid: {getattr(e, 'request', None) and e.request.url}")


def parse_pdf_text(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()

        doc.close()

        if not text.strip():
            raise UnprocessableContentError("PDF contains no extractable text.")
        return text
    except Exception as e:
        raise UnprocessableContentError(f"File is corrupt or not a valid PDF: {e}")



async def extract_skills_from_text(text: str) -> dict:
    tech_prompt = f"""
    You are a precise data extraction robot. Your task is to identify and list technical skills from the provided CV text.
    The CV text can be in English or Turkish. Extract the skills in English.

    Instructions:
    - Scan the ENTIRE CV text, including the summary, work history, and projects, not just a dedicated skills list.
    - Look for programming languages, databases, frameworks, libraries, algorithms, tools, and technical concepts.
    - Do NOT include skill levels (like 'Orta', 'Beginner').
    - Do NOT include category headers (like 'Programlama Dilleri:').
    - Do NOT include human languages (like 'English', 'Turkish').
    - Respond ONLY with a single, clean, comma-separated list.

    CV Text:
    ---
    {text}
    ---
    """

    soft_prompt = f"""
    You are an expert HR analyst. Your task is to INFER the candidate's soft skills from the descriptions of their actions, roles, and experiences in the CV text.

    The CV text can be in English or Turkish.
    
    Instructions:
    - Read the summary and work history sections carefully.
    - Based on the actions described, infer the relevant soft skill. For example:
    - If the text says "collaborated with team members" or "worked in a team", infer "Teamwork" and "Collaboration".
    - If the text says "presented results to stakeholders" or "gained experience in customer communication", infer "Communication" and "Presentation Skills".
    - If the text says "developed a new solution for a problem", infer "Problem Solving".
    - Respond ONLY with a single, clean, comma-separated list of the skills you inferred.
    - If the text contains no descriptions of actions from which to infer skills, respond with ONLY the word "NONE".


    CV Text:
    ---
    {text}
    ---
    """

    tech_skills_str = ""
    soft_skills_str = ""


    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            tech_response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": EXTRACTION_MODEL, "prompt": tech_prompt, "stream": False}
            )
            tech_response.raise_for_status()
            tech_skills_str = tech_response.json().get("response", "").strip()

            soft_response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": EXTRACTION_MODEL, "prompt": soft_prompt, "stream": False}
            )
            soft_response.raise_for_status()
            soft_skills_str = soft_response.json().get("response", "").strip()

        except httpx.RequestError as e:
            raise Exception(f"Failed to communicate with Ollama for skills extraction: {e}")

    tech_skills = [skill.strip() for skill in tech_skills_str.split(',') if skill.strip()]

    if soft_skills_str.upper() == 'NONE':
        soft_skills = []
    else:
        soft_skills = [skill.strip() for skill in soft_skills_str.split(',') if skill.strip()]

    return {"tech_skills": tech_skills, "soft_skills": soft_skills}

async def generate_embedding_from_text(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={
                    "model": EMBEDDING_MODEL,
                    "prompt": text,
                    "options": { "num_ctx": 2048 }
                }
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except httpx.RequestError as e:
            raise Exception(f"Failed to communicate with Ollama for embedding: {e}")

async def ensure_model_is_pulled(model_name: str):
    """Checks if a model is available locally and pulls it if not."""
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])

            if not any(m["name"].startswith(model_name) for m in models):
                print(f"Model '{model_name}' not found locally. Pulling from Ollama Hub...")

                pull_payload = {"name": model_name, "stream": False}
                async with httpx.AsyncClient(timeout=None) as pull_client:
                    async with pull_client.stream("POST", f"{OLLAMA_HOST}/api/pull", json=pull_payload) as pull_response:
                        pull_response.raise_for_status()
                        async for _ in pull_response.aiter_bytes():
                            pass

                print(f"Successfully pulled model '{model_name}'.")
            else:
                print(f"Model '{model_name}' is already available.")

    except Exception as e:
        print(f"Failed to ensure model '{model_name}'. Error: {e}")

async def generate_posting_vector(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={
                    "model": EMBEDDING_MODEL,
                    "prompt": text,
                    #"options": {"num_ctx": 2048}
                }
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except httpx.RequestError as e:
            raise Exception(f"Failed to communicate with Ollama for embedding: {e}")


async def analyze_match(cv_text: str, posting_text: str) -> str:
    prompt = f"""
        You are a professional HR analyst. Your task is to analyze the provided CV and Job Posting and write a structured analysis.

        IMPORTANT INSTRUCTIONS:
        1.  The input texts (CV and Job Posting) can be in Turkish or English.
        2.  Your entire analysis and output text MUST be in English.
        3.  You must structure your response with the following three headings: "Strengths:", "Gaps:", and "Summary:".
        4.  Under "Strengths:", create a bulleted list of matches between the CV and the job posting.
        5.  Under "Gaps:", create a bulleted list of key requirements from the posting that are missing from the CV.
        6.  Under "Summary:", write a final, 2-3 sentence professional conclusion about the candidate's suitability.
        7.  Provide ONLY this structured text. Do not add any other conversational text.

         --- 
         CV 
         --- 
         {cv_text} 
         --- 
         JOB POSTING 
         --- 
         {posting_text}
        """

    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
           # opts = {}
           # if EXTRACTION_MODEL.startswith("phi3"):
              #  opts["num_ctx"] = 4096  # phi3 için güvenli bağlam uzunluğu
               # opts["num_predict"] = 128

            payload = {
                "model": EXTRACTION_MODEL,
                "messages": [{'role': 'user', 'content': prompt}],
                "stream": False,
                #"options": opts,
            }

            response = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
            response.raise_for_status()

            summary_text = response.json().get("message", {}).get("content", "").strip()

            if not summary_text:
                raise Exception("Ollama returned an empty summary.")

            return summary_text

        except httpx.RequestError as e:
            raise Exception(f"Failed to communicate with Ollama for match analysis: {e}")

async def generate_embedding_from_postings(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_HOST}/api/embed",
                json={
                    "model": EMBEDDING_MODEL,
                    "input": texts,
                    #"options": { "num_ctx": 2048 }
                }
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except httpx.RequestError as e:
            raise Exception(f"Failed to communicate with Ollama for embedding: {e}")

async def extract_cv_info(cv_text: str) -> dict:

    prompt = f"""
    You are a highly skilled data extraction AI specializing in parsing human resources documents. Your task is to meticulously analyze the curriculum vitae (CV) provided below and extract key information, select the appropriate carrier_field from the fields given below.
    "Software Development", "Quality Assurance", "Product Management", "Business Development", "Design", "Product Management";
    "Operations", "Customer Education";
    "Sales", "Sales Operations";
    "Finance & Business Support";
    "Marketing and Communications", "Marketing Design";
    "Human Resources";
    "MindBehind";
    "CEO's Executive Office";
    
    **Your output MUST be a single, valid JSON object.** Do not include any text outside of the JSON object.

    If a specific piece of information cannot be found, you MUST use the string "unknown" as the value for that field.

    **JSON Schema to follow:**
    
        "first_name": "string",
        "last_name": "string",
        "email": "string",
        "phone": "string",
        "career_field": "string"
    

    ---CV TEXT---
    {cv_text}
    ---CV TEXT ENDS---
    """
    payload = {
        "model": EXTRACTION_MODEL,
        "messages": [{'role': 'user', 'content': prompt}],
        "stream": False,
        "format": "json"
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
            response.raise_for_status()
            result = response.json().get("message", {}).get("content", "").strip()

            if not result:
                raise ValueError("Ollama returned an empty content string.")

            parsed_data_from_ai = json.loads(result)

            career_field = parsed_data_from_ai.get("career_field", "unknown")
            department = get_department_for_career_field(career_field)

            parsed_data_from_ai["department"] = department

            return parsed_data_from_ai

        except json.JSONDecodeError:
            raise ValueError(f"Ollama returned a malformed JSON. Raw response: {result}")
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to communicate with Ollama: {e}")

def get_department_for_career_field(career_field: str) -> str:

    department_mapping = {
        "Engineering & Product": ["Software Development", "Quality Assurance", "Product Management",
                                  "Business Development",
                                  "Design"],
        "Partner Onboarding & Support": ["Operations", "Customer Education"],
        "Business Customer Success": ["Sales", "Sales Operations"],
        "Finance": ["Finance & Business Support"],
        "Marketing": ["Marketing and Communications", "Marketing Design"],
        "People & Culture": ["Human Resources"],
        "MindBehind": ["MindBehind"],
        "Ceo’s Office": ["CEO's Executive Office"]
    }
    for department, fields in department_mapping.items():
        if career_field in fields:
            return department

    return "Unknown Department"