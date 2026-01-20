"""
FastAPI backend for Resume ATS Analysis
========================================

Exposes the RAG pipeline as a RESTful API.

Endpoints:
- POST /analyze-resume: Analyze a resume PDF against a job description
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
from .rag_chain import analyze_resume
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


class AnalysisResponse(BaseModel):
    """Resume analysis response"""
    extracted_skills: list
    ats_result: Dict[str, Any]


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


@app.post("/analyze-resume", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_resume_endpoint(
    resume: UploadFile = File(..., description="Resume PDF file"),
    job_description: str = Form(..., description="Job description text")
) -> Dict[str, Any]:
    """
    Analyze a resume against a job description.
    
    Process:
    1. Read uploaded PDF file into bytes (in-memory, no disk storage)
    2. Extract text from PDF
    3. Run RAG analysis against job description
    4. Return structured results
    
    Args:
        resume: Uploaded PDF file
        job_description: Job description to compare against
    
    Returns:
        Dict with extracted_skills and ats_result
    
    Raises:
        HTTPException: If PDF extraction fails or analysis errors occur
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
        
        # Run RAG analysis
        try:
            result = analyze_resume(resume_text, job_description)
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
    print("  - POST /analyze-resume (Resume analysis)")
    print("  - GET /health (Health check)")
    print("  - GET /docs (API documentation)")
    print("=" * 70)


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    import uvicorn
    
    # Run with: python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
