import os, json
from langchain_google_genai import ChatGoogleGenerativeAI
# Importing LangChain's wrapper for using Gemini as LLM
# This will allow us to call Gemini Models
from retriever import load_retriever
# load_retriever() loads your FAISS index and returns a retriever that contains relevant chunks

from dotenv import load_dotenv
load_dotenv()
# Loads the .env file so GOOGLE_API_KEY becomes availabke to the program

# Building RAG CHAIN FUNCTION
def build_rag_chain():
    """
    Creates the full RAG pipeline:
    retriever -> prompt builder -> Gemini -> answer
    """
    # Here, we are building entire RAG system
    # It returns a function you can call to ask questions

    retriever = load_retriever()
    # Calls load_retriever()
    # This gives access to FAISS vectorstore, embedded chunks
    # This retriever fetches the top-k most relevant chunks for every query.

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    # ChatGoogleGenerativeAI creates a Gemini model object using LangChain
    # model: which Gemini LLM to use
    # temperature set to 0.2: low randomness = more factual answers
    # google_api_key: pulled from .env file

    # this llm object will be used to answer questions using retrieved context

    ### SKILL EXTRACTION FUNCTION
    def extract_skills_from_resume(resume_content: str):
        """
        Extract explicitly mentioned skills from resume content.
        Returns a list of clean, standardized skill names.
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

    ### ATS EVALUATION FUNCTION
    def evaluate_ats(resume_content: str, job_description: str, extracted_skills: list):
        """
        Evaluate resume against job description using ATS criteria.
        Uses extracted skills as context for the evaluation.
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

    ### RAG FUNCTION
    def rag_ask(question: str):
        """
        Run the complete RAG flow with skill extraction and ATS evaluation.
        Returns combined results: extracted skills + ATS evaluation.
        """
        docs = retriever.invoke(question)
        # Sends the question to FAISS
        # FAISS embeds the question[converting it into numerical vector] -> compares it with your chunk vectors -> returns top-k chunks
        # docs will be the list of LangChain Document objects [doc.page_content and doc.metadata]
        
        context = "\n\n".join([d.page_content for d in docs])
        # Here, We are extracting text from each retrieved chunk[which is in Document object form]
        # Joins them with two newlines to form a readable block
        # This becomes the "source of truth" for the model

        # STEP 1: Extract skills from resume content
        extracted_skills = extract_skills_from_resume(context)
        
        # STEP 2: Evaluate ATS with extracted skills as context
        ats_result = evaluate_ats(context, question, extracted_skills)
        
        # STEP 3: Combine both results into final output
        final_output = {
            "extracted_skills": extracted_skills,
            "ats_result": ats_result
        }
        
        return final_output
    
    return rag_ask
    # build_rag_chain() returns the function that performs a full RAG lookup + skill extraction + ATS evaluation
