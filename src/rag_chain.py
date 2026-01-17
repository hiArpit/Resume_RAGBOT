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

    ### RAG FUNCTION
    def rag_ask(question: str):
        # This function will run the entire RAG flow for a single question
        docs = retriever.invoke(question)
        # Sends the question to FAISS
        # FAISS embeds the question[converting it into numerical vector] -> compares it with your chunk vectors -> returns top-k chunks
        # docs will be the list of LangChain Document objects [doc.page_content and doc.metadata]
        
        context = "\n\n".join([d.page_content for d in docs])
        # Here, We are extracting text from each retrieved chunk[which is in Document object form]
        # Joins them with two newlines to form a readable block
        # This becomes the "source of truth" for the model

        ### Building the final prompted question
        prompt = f"""
You are an ATS (Applicant Tracking System) evaluator.

Analyze the resume content against the job description and return ONLY valid JSON.

Resume Content:
{context}

Job Description:
{question}

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


        # RAG Prompt is defined
        # It sets the persona (Eldoria lore expert)
        # It strictly tells the model to use only the supplied context.
        # It inserts your retrieved chunks
        # Finally adds the user's actual question

        response = llm.invoke(prompt)
        # Passes the prompt into Gemini
        # .invoke() will return a LangChain message object named response
        
        # Clean the response: remove markdown code blocks if present
        raw_response = response.content.strip()
        if raw_response.startswith("```"):
            # Remove markdown code block wrapper
            raw_response = raw_response.split("```")[1]  # Get content between code blocks
            if raw_response.startswith("json\n"):
                raw_response = raw_response[5:]  # Remove 'json\n' prefix
        
        try :
            parsed_output = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed_output = {
                "error": "Invalid JSON response from LLM",
                "raw_output": raw_response
            }

        return parsed_output
        # Retrieving the text portion of the LLM response
    
    return rag_ask
    # build_rag_chain() returns the function that performs a full RAG lookup + LLM response/answer
