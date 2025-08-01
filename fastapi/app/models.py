from pydantic import BaseModel, HttpUrl
from typing import List, Literal


class UnsupportedMediaTypeError(Exception):
    pass

class UnprocessableContentError(Exception):
    pass

# --- Pydantic Models ---
class ResumeParseRequest(BaseModel):
    resume: HttpUrl

class SuccessResponse(BaseModel):
    status: Literal["SUCCESS"] = "SUCCESS"
    soft_skills: List[str]
    tech_skills: List[str]
    parsed_cv: List[float]

class ErrorResponse(BaseModel):
    status: Literal["VALIDATION_ERROR", "ERROR"]
    errorCode: str
    message: str