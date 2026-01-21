import os, json
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()


# ==================== BACKEND API FUNCTIONS ====================

def extract_skills_only(resume_text: str) -> Dict[str, Any]:
    """
    Extract skills from resume text only.
    
    Performs skill extraction on raw resume text without requiring job description.
    Uses in-memory FAISS vector store for optimal chunking and retrieval.
    
    Args:
        resume_text: Raw resume text (string, can be multi-line)
    
    Returns:
        Dictionary with:
        - "extracted_skills": List of skill strings
    
    Raises:
        ValueError: If resume text is empty or API key not set
    
    Example:
        result = extract_skills_only(resume_text)
        skills = result["extracted_skills"]  # ["Python", "AWS", ...]
    """
    from .chunker import make_chunks
    from .vector_store import build_vector_store
    
    # Validate input
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")
    
    # STEP 1: Chunk the resume text
    chunks = make_chunks([resume_text])
    
    # STEP 2: Build in-memory FAISS vector store (no disk persistence)
    vectordb = build_vector_store(chunks, persist_directory=None)
    
    # STEP 3: Retrieve relevant chunks
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke("skills experience")
    context = "\n\n".join([d.page_content for d in docs])
    
    # STEP 4: Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        google_api_key=api_key
    )
    
    # STEP 5: Extract skills using existing helper
    extracted_skills = _extract_skills_from_resume(llm, context)
    
    return {"extracted_skills": extracted_skills}


def evaluate_ats_only(resume_text: str, job_description: str) -> Dict[str, Any]:
    """
    Evaluate ATS score of resume against job description.
    
    Performs ATS evaluation on raw resume text and job description.
    Uses in-memory FAISS vector store for optimal chunking and retrieval.
    
    Args:
        resume_text: Raw resume text (string, can be multi-line)
        job_description: Job description to compare against
    
    Returns:
        Dictionary with ATS scores and recommendations:
        - "ats_score": Overall score (0-100)
        - "skills_match_score": Score for skills match (out of 40)
        - "experience_relevance_score": Score for experience (out of 30)
        - "tools_keywords_score": Score for tools & keywords (out of 20)
        - "resume_clarity_score": Score for clarity (out of 10)
        - "missing_skills": List of missing skills
        - "weak_areas": List of weak areas
        - "suggestions": List of improvement suggestions
    
    Raises:
        ValueError: If inputs are empty or API key not set
    
    Example:
        result = evaluate_ats_only(resume_text, job_description)
        score = result["ats_score"]  # 75
    """
    from .chunker import make_chunks
    from .vector_store import build_vector_store
    
    # Validate inputs
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty")
    if not job_description or not job_description.strip():
        raise ValueError("Job description cannot be empty")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")
    
    # STEP 1: Chunk the resume text
    chunks = make_chunks([resume_text])
    
    # STEP 2: Build in-memory FAISS vector store (no disk persistence)
    vectordb = build_vector_store(chunks, persist_directory=None)
    
    # STEP 3: Retrieve relevant chunks based on job description
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(job_description)
    context = "\n\n".join([d.page_content for d in docs])
    
    # STEP 4: Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        google_api_key=api_key
    )
    
    # STEP 5: Evaluate ATS using existing helper (no skill extraction)
    ats_result = _evaluate_ats(llm, context, job_description, extracted_skills=[])
    
    return ats_result


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
    from .retriever import load_retriever
    
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
