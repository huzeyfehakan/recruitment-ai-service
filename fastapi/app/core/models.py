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

class PostingsRequest(BaseModel):
    postings: List[str]




# ---------- Success Responses ----------
class CVInfo(BaseModel):
    """Represents the basic contact and career info extracted from a CV."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    career_field: Optional[str] = None
    department: Optional[str] = None

class ParsedResumeSuccess(BaseModel):
    """Represents the successful result of parsing a resume."""
    status: constr(pattern="^SUCCESS$") = "SUCCESS"
    soft_skills: List[str]
    tech_skills: List[str]
    parsed_cv_vector: List[float]
    parsed_cv_text: str
    cv_info: CVInfo



class PostingSuccess(BaseModel):
    status: constr(pattern="^SUCCESS$") = "SUCCESS"
    posting_vector: List[float]

class MatchAnalyzeSuccess(BaseModel):
    status: constr(pattern="^SUCCESS$") = "SUCCESS"
    result: str

class PostingsSuccess(BaseModel):
    status: constr(pattern="^SUCCESS$") = "SUCCESS"
    posting_vectors: List[List[float]]

# ---------- Error Response ----------
class ErrorResponse(BaseModel):
    status: constr(pattern="^(VALIDATION_ERROR|ERROR)$")
    errorCode: str
    message: str


