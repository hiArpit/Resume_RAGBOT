"""
FastAPI backend for Resume ATS Analysis
========================================

Exposes the RAG pipeline as a RESTful API.

Endpoints:
- POST /extract-skills: Extract skills from a resume PDF
- POST /evaluate-ats: Evaluate ATS score against a job description
- GET /health: Health check endpoint
"""

import os
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import core RAG functions (relative imports)
from .rag_chain import extract_skills_only, evaluate_ats_only
from .pdf_loader import extract_text_from_pdf_bytes



# ==================== FASTAPI APP INITIALIZATION ====================

app = FastAPI(
    title="Resume ATS Analyzer API",
    description="RAG-based resume analysis against job descriptions",
    version="1.0.0"
)


# ==================== RESPONSE MODELS ====================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str


class SkillsResponse(BaseModel):
    """Skills extraction response"""
    extracted_skills: list


class AtsResponse(BaseModel):
    """ATS evaluation response"""
    ats_score: int
    skills_match_score: int
    experience_relevance_score: int
    tools_keywords_score: int
    resume_clarity_score: int
    missing_skills: list
    weak_areas: list
    suggestions: list


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    details: str = ""


# ==================== ENDPOINTS ====================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status of the API
    """
    return HealthResponse(
        status="healthy",
        service="Resume ATS Analyzer API",
        version="1.0.0"
    )


@app.post("/extract-skills", response_model=SkillsResponse, tags=["Analysis"])
async def extract_skills_endpoint(
    resume: UploadFile = File(..., description="Resume PDF file")
) -> Dict[str, Any]:
    """
    Extract skills from a resume.
    
    Process:
    1. Read uploaded PDF file into bytes (in-memory, no disk storage)
    2. Extract text from PDF
    3. Extract skills using RAG pipeline
    4. Return extracted skills only
    
    Args:
        resume: Uploaded PDF file
    
    Returns:
        Dict with extracted_skills list
    
    Raises:
        HTTPException: If PDF extraction fails or analysis errors occur
    """
    
    try:
        # Read PDF file bytes (in-memory, no disk I/O)
        pdf_bytes = await resume.read()
        
        if not pdf_bytes:
            raise HTTPException(
                status_code=400,
                detail="Resume file is empty"
            )
        
        # Extract text from PDF
        try:
            resume_text = extract_text_from_pdf_bytes(pdf_bytes)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        
        # Extract skills
        try:
            result = extract_skills_only(resume_text)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        
        return result
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/evaluate-ats", response_model=AtsResponse, tags=["Analysis"])
async def evaluate_ats_endpoint(
    resume: UploadFile = File(..., description="Resume PDF file"),
    job_description: str = Form(..., description="Job description text")
) -> Dict[str, Any]:
    """
    Evaluate ATS score of a resume against a job description.
    
    Process:
    1. Read uploaded PDF file into bytes (in-memory, no disk storage)
    2. Extract text from PDF
    3. Run ATS evaluation against job description
    4. Return ATS scores and recommendations
    
    Args:
        resume: Uploaded PDF file
        job_description: Job description to compare against
    
    Returns:
        Dict with ATS scores, missing skills, weak areas, and suggestions
    
    Raises:
        HTTPException: If PDF extraction fails, inputs are invalid, or analysis errors occur
    """
    
    # Validate job description
    if not job_description or not job_description.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty"
        )
    
    try:
        # Read PDF file bytes (in-memory, no disk I/O)
        pdf_bytes = await resume.read()
        
        if not pdf_bytes:
            raise HTTPException(
                status_code=400,
                detail="Resume file is empty"
            )
        
        # Extract text from PDF
        try:
            resume_text = extract_text_from_pdf_bytes(pdf_bytes)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        
        # Evaluate ATS
        try:
            result = evaluate_ats_only(resume_text, job_description)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        
        return result
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with consistent error format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "Analysis failed",
            "details": exc.detail
        }
    )


# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    print("=" * 70)
    print("Resume ATS Analyzer API - Starting")
    print("=" * 70)
    print("✓ FastAPI application initialized")
    print("✓ Endpoints available:")
    print("  - POST /extract-skills (Extract skills from resume)")
    print("  - POST /evaluate-ats (Evaluate ATS score)")
    print("  - GET /health (Health check)")
    print("  - GET /docs (API documentation)")
    print("=" * 70)


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.getenv("PORT", 8000))
    # Run with: python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
