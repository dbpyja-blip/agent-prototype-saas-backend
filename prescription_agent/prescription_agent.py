
import os
import fitz  # PyMuPDF
import requests
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# CrewAI & Tooling
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# Load Environment Variables
load_dotenv()

app = FastAPI(title="Consultation Agent - Production Grade")

from fastapi.middleware.cors import CORSMiddleware

# allow_origins=["*"] is incompatible with allow_credentials=True per the CORS spec.
# Browsers will refuse the response if both are set simultaneously because a wildcard
# origin cannot be used when credentials (cookies / auth headers) are included.
# Solution: list every trusted front-end origin explicitly so credentials work correctly,
# and fall back to the wildcard-without-credentials pattern for any other caller.
ALLOWED_ORIGINS = [
    "https://healthcareagents.dimensionleap.com",
    # add more origins here if needed, e.g. "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # explicit list — required when credentials=True
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# --- 1. LLM CONFIGURATION ---
# Using gpt-4o for complex reasoning agents to reduce hallucinations
primary_llm = LLM(
    model=f"azure/{os.getenv('AZURE_LLM_DEPLOYMENT')}",
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION"),
    temperature=0.0
)
# --- 2. ENHANCED PRODUCTION TOOLS ---
@tool("OpenFDA_Deep_Search")
def deep_openfda_tool(drug_name: str) -> str:
    """
    Normalizes drug names and performs an exhaustive search on OpenFDA.
    Strips dosages and prefixes to ensure API hits.
    """
    import re
    # 1. CLEANING: Remove 'Tab.', 'Cap.', 'Syp.' and dosages like '20mg' or '800/160'
    clean_name = re.sub(r'^(Tab\.|Syp\.|Inj\.|Cap\.|Tab|Syp|Inj|Cap)\s+', '', drug_name, flags=re.IGNORECASE)
    clean_name = re.sub(r'\d+\s*(mg|g|mcg|ml|l|units)\b', '', clean_name, flags=re.IGNORECASE)
    clean_name = re.sub(r'[\d\(\)/]', '', clean_name).strip()
    
    base_url = "https://api.fda.gov/drug/label.json"
    api_key = os.getenv("OPENFDA_API_KEY")
    
    # 2. SEARCH STRATEGY: Try Generic Name first, then Brand
    search_queries = [
        f'openfda.generic_name:"{clean_name}"',
        f'openfda.brand_name:"{clean_name}"'
    ]
    
    for query in search_queries:
        params = {'search': query, 'limit': 1, 'api_key': api_key}
        try:
            response = requests.get(base_url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "results" in data:
                    res = data["results"][0]
                    # Return the official interaction section
                    return res.get("drug_interactions", ["Warning: Interaction section missing in FDA label."])[0]
        except Exception:
            continue
            
    return f"Warning: No official FDA records found for '{clean_name}'. Manual review required."

# --- 3. SCHEMAS (Guardrails) ---
class MedicationDetail(BaseModel):
    name: str
    dosage: str
    timing: Optional[str] = None
    doc_noted_warnings: Optional[str] = Field(None, description="STRICTLY ONLY handwritten notes from the PDF.")
    fda_risk_assessment: str = Field(..., description="Findings from the OpenFDA tool.")

class SafetyReport(BaseModel):
    verdict: str = Field(description="SAFE, WARNING, or DANGEROUS")
    medications: List[MedicationDetail]
    interaction_summary: str
    compliance_check: str = Field(description="Audit of document instructions vs safety data.")
    disclaimer: str = "OFFICIAL: This is an AI prototype. Validation by a licensed MD/Pharmacist is mandatory."

# --- 5. PDF PROCESSING ---
def extract_text_from_pdf(file_bytes):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        # Using blocks keeps the association between a drug name and the note below it
        return "\\n".join([page.get_text("blocks")[i][4] for page in doc for i in range(len(page.get_text("blocks")))])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF Parsing Error: {str(e)}")

# --- 6. CORE BUSINESS LOGIC (Stateless) ---
def run_analysis(raw_text: str):
    # --- AGENT INITIALIZATION (STATELESS) ---
    extractor = Agent(
        role='Clinical Registrar',
        goal='Extract medications and dosages from {text}. Capture handwritten notes ONLY.',
        backstory="""You are a meticulous medical scribe. 
        STRICT RULE: If a medication has no handwritten 'Note' or 'Interaction Check' 
        written next to it in the PDF, you MUST set 'doc_noted_warnings' to None. 
        DO NOT use your own medical knowledge to fill this field.""",
        llm=primary_llm,
        verbose=True
    )

    pharmacist = Agent(
        role='Senior Clinical Pharmacist',
        goal='Cross-verify drugs using the OpenFDA tool. Do not hallucinate cross-case data.',
        backstory="""You provide pharmacological assessments grounded in FDA data. 
        You must only report risks for the drugs provided in the current session. 
        Never reference 'Bleeding Risk' unless the drug is a known anticoagulant.""",
        tools=[deep_openfda_tool],
        llm=primary_llm,
        verbose=True
    )

    verifier = Agent(
        role='Safety Compliance Officer',
        goal='Finalize the SafetyReport. Ensure Doctor Notes are not hallucinated.',
        backstory="""You resolve conflicts. If 'doc_noted_warnings' is None, the final 
        'Doctor's Note' field must be empty. Ensure 'Bleeding Risk' labels from previous 
        unrelated cases are not misapplied here.""",
        llm=primary_llm,
        verbose=True
    )

    t1 = Task(
        description=f"Identify medications in: {raw_text}. Capture associated handwritten notes or set to None.",
        expected_output="A list of current medications and their specific document warnings.",
        agent=extractor
    )

    t2 = Task(
        description="Run deep FDA search for the current drugs. Focus on interactions and contraindications.",
        expected_output="An audit of drug interactions based on FDA labels.",
        agent=pharmacist,
        context=[t1]
    )

    t3 = Task(
        description="Verify findings. If 'doc_noted_warnings' is None, do not include it. Output SafetyReport.",
        expected_output="Final validated SafetyReport JSON.",
        agent=verifier,
        context=[t1, t2],
        output_pydantic=SafetyReport
    )

    # Crew is initialized fresh per request to prevent context leakage
    crew = Crew(
        agents=[extractor, pharmacist, verifier],
        tasks=[t1, t2, t3],
        process=Process.sequential
    )

    result = crew.kickoff()
    
    # Ensure we return the Pydantic model
    if hasattr(result, 'pydantic') and result.pydantic:
        return result.pydantic
    elif hasattr(result, 'json_dict') and result.json_dict:
        return SafetyReport(**result.json_dict)
    else:
        # Fallback if parsing failed but we have text (shouldn't happen with output_pydantic)
        raise ValueError(f"Failed to generate structured SafetyReport. Raw: {result.raw}")

# --- 7. API ENDPOINT ---
@app.post("/process-prescription", response_model=SafetyReport)
async def process_prescription(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")
    
    content = await file.read()
    raw_text = extract_text_from_pdf(content)
    return run_analysis(raw_text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
