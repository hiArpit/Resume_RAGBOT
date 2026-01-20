import os, json
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()


# ==================== CORE REUSABLE RAG FUNCTION ====================

def analyze_resume(resume_text: str, job_description: str) -> Dict[str, Any]:
    """
    Core RAG pipeline for resume ATS analysis.
    
    Takes raw resume text and job description, performs complete analysis:
    - Chunks the resume text (using chunker.make_chunks)
    - Builds in-memory FAISS vector store (using vector_store.build_vector_store)
    - Retrieves relevant chunks based on job description
    - Extracts skills from retrieved chunks
    - Evaluates ATS score against job description
    
    Args:
        resume_text: Raw resume text (string, can be multi-line)
        job_description: Job description to compare against
    
    Returns:
        Dictionary with:
        - "extracted_skills": List of skill strings
        - "ats_result": Dict with scores and recommendations
    
    Raises:
        ValueError: If inputs are empty or API key not set
    """
    from chunker import make_chunks
    from vector_store import build_vector_store
    
    # Validate inputs
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty")
    if not job_description or not job_description.strip():
        raise ValueError("Job description cannot be empty")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")
    
    # STEP 1: Chunk the resume text using existing chunker
    # make_chunks expects List[str] of pages, so wrap single text as single page
    chunks = make_chunks([resume_text])
    
    # STEP 2: Build in-memory FAISS vector store (no disk persistence)
    # persist_directory=None keeps FAISS index in-memory only, prevents disk writes
    vectordb = build_vector_store(chunks, persist_directory=None)
    
    # STEP 3: Create retriever and retrieve relevant chunks
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(job_description)
    context = "\n\n".join([d.page_content for d in docs])
    
    # STEP 4: Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        google_api_key=api_key
    )
    
    # STEP 5: Extract skills from retrieved context
    extracted_skills = _extract_skills_from_resume(llm, context)
    
    # STEP 6: Evaluate ATS score
    ats_result = _evaluate_ats(llm, context, job_description, extracted_skills)
    
    # Return structured result (FAISS automatically garbage collected after this function)
    return {
        "extracted_skills": extracted_skills,
        "ats_result": ats_result
    }


def analyze_resume_input(resume_input, job_description: str, input_type: str = "text") -> Dict[str, Any]:
    """
    Analyze resume from dynamic input (PDF bytes or text string).
    
    Wrapper around analyze_resume() that handles both PDF and text input.
    Extracts text from PDF if needed, then runs the full RAG pipeline.
    
    Args:
        resume_input: Either:
            - bytes: Raw PDF file content
            - str: Resume text directly
        job_description: Job description to compare against
        input_type: Type of input, either "pdf" or "text" (default: "text")
    
    Returns:
        Dictionary with:
        - "extracted_skills": List of skill strings
        - "ats_result": Dict with scores and recommendations
    
    Raises:
        ValueError: If input type is invalid, text extraction fails, or inputs are empty
    
    Examples:
        # From PDF bytes (uploaded file)
        pdf_bytes = request.files['resume'].read()
        result = analyze_resume_input(pdf_bytes, job_description, input_type="pdf")
        
        # From text string
        result = analyze_resume_input(resume_text, job_description, input_type="text")
    """
    from pdf_loader import extract_text_from_pdf_bytes
    
    # Validate input_type
    if input_type not in ("pdf", "text"):
        raise ValueError(f"input_type must be 'pdf' or 'text', got '{input_type}'")
    
    # Extract resume text based on input type
    if input_type == "pdf":
        if not isinstance(resume_input, bytes):
            raise ValueError("For input_type='pdf', resume_input must be bytes")
        resume_text = extract_text_from_pdf_bytes(resume_input)
    
    elif input_type == "text":
        if not isinstance(resume_input, str):
            raise ValueError("For input_type='text', resume_input must be string")
        resume_text = resume_input
    
    # Run core RAG analysis with extracted text
    return analyze_resume(resume_text, job_description)


# ==================== HELPER FUNCTIONS ====================

def _extract_skills_from_resume(llm: ChatGoogleGenerativeAI, resume_content: str) -> list:
    """
    Extract explicitly mentioned skills from resume content.
    
    Args:
        llm: ChatGoogleGenerativeAI instance
        resume_content: Resume text to analyze
    
    Returns:
        List of skill strings
    """
    skill_extraction_prompt = f"""
You are a resume skill extractor. Analyze the resume content and extract ONLY the explicitly mentioned skills.

Resume Content:
{resume_content}

STRICT OUTPUT FORMAT (JSON ONLY — no markdown, no explanation):

{{
  "skills": [string]
}}

Rules:
- Extract ONLY skills that are explicitly mentioned in the resume
- DO NOT infer or guess skills
- Return clean, standardized skill names (e.g., "Python", "Project Management")
- Remove duplicates
- If no skills are found, return empty array
"""
    response = llm.invoke(skill_extraction_prompt)
    raw_response = response.content.strip()
    
    # Clean markdown code blocks if present
    if raw_response.startswith("```"):
        raw_response = raw_response.split("```")[1]
        if raw_response.startswith("json\n"):
            raw_response = raw_response[5:]
    
    try:
        parsed_skills = json.loads(raw_response)
        return parsed_skills.get("skills", [])
    except json.JSONDecodeError:
        return []


def _evaluate_ats(llm: ChatGoogleGenerativeAI, resume_content: str, job_description: str, extracted_skills: list) -> Dict[str, Any]:
    """
    Evaluate resume against job description using ATS criteria.
    
    Args:
        llm: ChatGoogleGenerativeAI instance
        resume_content: Resume text
        job_description: Job description
        extracted_skills: List of skills already extracted
    
    Returns:
        Dict with ATS scores and recommendations
    """
    # Include extracted skills in the prompt context
    skills_context = f"\n\nExtracted Skills:\n{', '.join(extracted_skills)}" if extracted_skills else ""
    
    ats_prompt = f"""
You are an ATS (Applicant Tracking System) evaluator.

Analyze the resume content against the job description and return ONLY valid JSON.

Resume Content:
{resume_content}
{skills_context}

Job Description:
{job_description}

Scoring Rules:
- Skills match: 40%
- Experience relevance: 30%
- Tools & keywords: 20%
- Resume clarity & impact: 10%

STRICT OUTPUT FORMAT (JSON ONLY — no markdown, no explanation):

{{
  "ats_score": number,
  "skills_match_score": number,
  "experience_relevance_score": number,
  "tools_keywords_score": number,
  "resume_clarity_score": number,
  "missing_skills": [string],
  "weak_areas": [string],
  "suggestions": [string]
}}

Rules:
- Scores must be integers
- Total ats_score must be out of 100
- Do NOT use external knowledge
- If data is missing, mark it clearly
"""
    response = llm.invoke(ats_prompt)
    raw_response = response.content.strip()
    
    # Clean markdown code blocks if present
    if raw_response.startswith("```"):
        raw_response = raw_response.split("```")[1]
        if raw_response.startswith("json\n"):
            raw_response = raw_response[5:]
    
    try:
        parsed_output = json.loads(raw_response)
    except json.JSONDecodeError:
        parsed_output = {
            "error": "Invalid JSON response from LLM",
            "raw_output": raw_response
        }
    
    return parsed_output


# ==================== LEGACY CLI SUPPORT ====================

def build_rag_chain():
    """
    Creates the full RAG pipeline for CLI compatibility.
    Uses pre-built FAISS index from load_retriever().
    
    Returns:
        Function that takes job_description and returns analysis dict
    """
    from retriever import load_retriever
    
    retriever = load_retriever()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        google_api_key=api_key
    )

    def rag_ask(question: str):
        """
        Run the complete RAG flow with skill extraction and ATS evaluation.
        Uses pre-built FAISS index from file.
        """
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        
        # Use core helper functions
        extracted_skills = _extract_skills_from_resume(llm, context)
        ats_result = _evaluate_ats(llm, context, question, extracted_skills)
        
        return {
            "extracted_skills": extracted_skills,
            "ats_result": ats_result
        }
    
    return rag_ask
