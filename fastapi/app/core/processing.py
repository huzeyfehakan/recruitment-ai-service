import os
import httpx
import fitz
import json
from io import BytesIO
from ..models import UnsupportedMediaTypeError, UnprocessableContentError

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

EMBEDDING_MODEL = "all-minilm"
EXTRACTION_MODEL = "gemma:2b"


async def download_and_validate_cv(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
        except httpx.RequestError as e:
            raise UnprocessableContentError(f"URL is invalid or unreachable: {e.request.url}")


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
    prompt = f"""
        You are an expert HR assistant. Your task is to extract technical skills (tech_skills) and soft skills (soft_skills) from the provided resume text and return them in JSON format.

        The resume text can be in English or Turkish. Analyze the text and extract the skills in their original language.

        Create a JSON output for the following CV text. Provide ONLY the JSON output and nothing else. Tech skills and Soft skills will be mapped in to a list object in Python. Give a proper response.

        CV Text:
        ---
        {text}
        ---
        """
    final_chunk = {}
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            payload = {
                "model": EXTRACTION_MODEL,
                "messages": [{'role': 'user', 'content': prompt}],
                "format": "json",
                "stream": False
            }
            async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=payload) as response:
                response.raise_for_status()
                # We process the stream line by line and only keep the last one
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        final_chunk = chunk # Keep overwriting until the last chunk

            # The full message is in the 'message' field of the final chunk
            raw_content = final_chunk.get("message", {}).get("content", "")

            if not raw_content:
                raise Exception("Ollama returned an empty response.")

            # Now we run our robust JSON extraction on the clean, final content
            json_start = raw_content.find('{')
            json_end = raw_content.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_string = raw_content[json_start:json_end]
                response_json = json.loads(json_string)
            else:
                response_json = {}

            return {
                "tech_skills": response_json.get("tech_skills", []),
                "soft_skills": response_json.get("soft_skills", [])
            }
        except (httpx.RequestError, json.JSONDecodeError) as e:
            raise Exception(f"Failed to communicate with or parse streaming response from Ollama: {e}")

async def generate_embedding_from_text(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={
                    "model": EMBEDDING_MODEL,
                    "prompt": text
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