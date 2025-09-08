from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from .core import processing
from .core.exceptions import (
    AppError,
    UnsupportedMediaTypeError,
    UnprocessableContentError,
    InvalidRequestError,
    EmptyStringError,
)
from .core.models import (
    ResumeParseRequest,
    ParsedResumeSuccess,
    ErrorResponse,
    PostingEmbeddingRequest,
    PostingSuccess,
    MatchAnalyzeRequest,
    MatchAnalyzeSuccess,
    PostingsRequest,
    PostingsSuccess,
    CVInfo
)

# This new 'lifespan' function will run on application startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run before the application starts accepting requests
    print("Application startup: Ensuring AI models are available...")
    await processing.ensure_model_is_pulled(processing.EXTRACTION_MODEL)
    await processing.ensure_model_is_pulled(processing.EMBEDDING_MODEL)
    print("Model check complete. Application is ready.")
    yield
    # Code to run on application shutdown
    print("Application shutdown.")

app = FastAPI(
    title="Recruitment AI Service",
    version="1.0.0",
    lifespan=lifespan
)

# ---- Unified error JSON helper ----
def error_json(status_code: int, error_code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "VALIDATION_ERROR" if status_code < 500 else "ERROR",
            "errorCode": error_code,
            "message": message,
        },
    )

# ---- Global Exception Handlers ----
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return error_json(exc.status_code, exc.error_code, exc.message)

@app.exception_handler(RequestValidationError)
async def req_validation_handler(request: Request, exc: RequestValidationError):
    return error_json(400, "INVALID_REQUEST", "given url is not valid")

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    code = f"HTTP_{exc.status_code}"
    msg = str(exc.detail or "unexpected error occured")
    status = 500 if exc.status_code >= 500 else exc.status_code
    return error_json(status, code, msg)

@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    # TODO: logger.exception("Unhandled", exc_info=exc)
    return error_json(500, "INTERNAL_SERVER_ERROR", "unexpected error occured")

# --- API Endpoint-1
@app.post("/api/v1/parsed-resumes", response_model=ParsedResumeSuccess, status_code=200)
async def parse_resume_endpoint(request: ResumeParseRequest):
    pdf_bytes = await processing.download_and_validate_cv(str(request.resume))
    cv_text = processing.parse_pdf_text(pdf_bytes)
    skills_data = await processing.extract_skills_from_text(cv_text)
    cv_vector = await processing.generate_embedding_from_text(cv_text)
    cv_info = await processing.extract_cv_info(cv_text)
    return ParsedResumeSuccess(
        soft_skills=skills_data["soft_skills"],
        tech_skills=skills_data["tech_skills"],
        parsed_cv_vector=cv_vector,
        parsed_cv_text = cv_text,
        cv_info=CVInfo(**cv_info),

    )

# --- API Endpoint-2
@app.post("/api/v1/posting", response_model=PostingSuccess, status_code=200)
async def posting_to_embedding_endpoint(request: PostingEmbeddingRequest):
    if not request.posting_string or not request.posting_string.strip():
        raise EmptyStringError("given string is not valid")

    posting_vector = await processing.generate_posting_vector(request.posting_string)
    return PostingSuccess(
        posting_vector=posting_vector
    )


# --- API Endpoint-3
@app.post("/api/v1/match-analyze", response_model=MatchAnalyzeSuccess, status_code=200)
async def match_analyze_endpoint(request: MatchAnalyzeRequest):
    if (not request.parsed_cv or not request.parsed_cv.strip()
            or not request.posting_string or not request.posting_string.strip()):
        return JSONResponse(
            status_code=400,
            content={
                "status": "VALIDATION_ERROR",
                "result": "cv_vector and job_posting_vector are required and cannot be empty",
            },
        )

    analyze_result = await processing.analyze_match(request.parsed_cv, request.posting_string)
    return MatchAnalyzeSuccess(
        result=analyze_result
    )
# --- API Endpoint-4
@app.post("/api/v1/posting-bulk", response_model=PostingsSuccess, status_code=200)
async def postings_to_embeddings_endpoint(request: PostingsRequest):
    for posting in request.postings:
        if not posting or not posting.strip():
            raise EmptyStringError("given string is not valid")

    posting_vectors = await processing.generate_embedding_from_postings(request.postings)
    return PostingsSuccess(
        posting_vectors=posting_vectors
    )