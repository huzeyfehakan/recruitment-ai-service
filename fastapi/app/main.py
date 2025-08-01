from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from .core import processing
from .models import (
    ResumeParseRequest, SuccessResponse, ErrorResponse,
    UnsupportedMediaTypeError, UnprocessableContentError
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


@app.exception_handler(UnsupportedMediaTypeError)
async def unsupported_media_type_handler(request: Request, exc: UnsupportedMediaTypeError):
    return JSONResponse(status_code=415,
                        content=ErrorResponse(status="VALIDATION_ERROR", errorCode="UNSUPPORTED_FILE_TYPE",
                                              message="only pdf files are supported").model_dump())


@app.exception_handler(UnprocessableContentError)
async def unprocessable_content_handler(request: Request, exc: UnprocessableContentError):
    return JSONResponse(status_code=422,
                        content=ErrorResponse(status="VALIDATION_ERROR", errorCode="UNPROCESSABLE_CONTENT",
                                              message=exc.message).model_dump())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content=ErrorResponse(status="VALIDATION_ERROR", errorCode="INVALID_REQUEST",
                                                               message="given url is not valid").model_dump())


# --- API Endpoint
@app.post("/api/v1/parsed-resumes", response_model=SuccessResponse, status_code=200)
async def parse_resume_endpoint(request: ResumeParseRequest):
    try:
        pdf_bytes = await processing.download_and_validate_cv(str(request.resume))
        cv_text = processing.parse_pdf_text(pdf_bytes)


        skills_data = await processing.extract_skills_from_text(cv_text)
        cv_vector = await processing.generate_embedding_from_text(cv_text)

        return SuccessResponse(
            soft_skills=skills_data["soft_skills"],
            tech_skills=skills_data["tech_skills"],
            parsed_cv=cv_vector
        )
    except (UnsupportedMediaTypeError, UnprocessableContentError) as e:
        raise e
    except Exception as e:
        # Daha detaylı hata loglaması
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred.")