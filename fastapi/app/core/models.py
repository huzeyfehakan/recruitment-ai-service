from pydantic import BaseModel, HttpUrl, constr
from typing import List, Optional

# ---------- Requests ----------
class ResumeParseRequest(BaseModel):

    resume: HttpUrl

class PostingEmbeddingRequest(BaseModel):

    posting_string: str

class MatchAnalyzeRequest(BaseModel):
    parsed_cv: str
    posting_string: str


# ---------- Success Responses ----------
class ParsedResumeSuccess(BaseModel):
    status: constr(pattern="^SUCCESS$") = "SUCCESS"
    soft_skills: List[str]
    tech_skills: List[str]
    parsed_cv_vector: List[float]
    parsed_cv_text: str


class PostingSuccess(BaseModel):
    status: constr(pattern="^SUCCESS$") = "SUCCESS"
    posting_vector: List[float]

class MatchAnalyzeSuccess(BaseModel):
    status: constr(pattern="^SUCCESS$") = "SUCCESS"
    result: str


# ---------- Error Response ----------
class ErrorResponse(BaseModel):
    status: constr(pattern="^(VALIDATION_ERROR|ERROR)$")
    errorCode: str
    message: str