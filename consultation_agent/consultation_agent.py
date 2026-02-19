import os
import re
import time
import uuid
import logging
import asyncio
from typing import List, Optional
from datetime import datetime

import tiktoken
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pythonjsonlogger import jsonlogger
from dotenv import load_dotenv

# CrewAI Imports
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()

# ==========================================
# 1. UPDATED MODELS FOR RAW INPUT
# ==========================================
class UnstructuredRequest(BaseModel):
    # Accepting raw text instead of a list of segments
    raw_text: str = Field(..., min_length=20, description="The raw, unlabelled transcript paragraph")
    request_id: Optional[str] = Field(None)

class SOAPNote(BaseModel):
    subjective: str = Field(..., alias="Subjective")
    objective: str = Field(..., alias="Objective")
    assessment: str = Field(..., alias="Assessment")
    plan: str = Field(..., alias="Plan")

    @field_validator('*', mode='before')
    @classmethod
    def enforce_not_documented(cls, v: str) -> str:
        if not v or str(v).strip().lower() in ["n/a", "none", "unknown", ""]:
            return "Not Documented"
        return str(v)

class ServiceResponse(BaseModel):
    success: bool
    data: Optional[SOAPNote] = None
    error: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

# ==========================================
# 2. ENHANCED AGENTIC ORCHESTRATOR
# ==========================================
class AgenticOrchestrator:
    def __init__(self):
        deployment = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4o-mini")

        self.llm = LLM(
            model=f"azure/{deployment}",
            api_key=os.getenv("AZURE_API_KEY"),
            endpoint=os.getenv("AZURE_ENDPOINT"), 
            api_version=os.getenv("AZURE_API_VERSION"),
            temperature=0.0, # Critical for medical consistency
            timeout=60
        )

        # The agent now assumes the role of a "Linguistic Auditor" 
        # specifically trained in speaker diarization logic.
        self.specialist = Agent(
            role="Clinical Diarization & Documentation Specialist",
            goal="Identify speakers in raw text and extract clinical SOAP data.",
            backstory=(
                "You are an expert at parsing medical conversations where speaker labels are missing. "
                "You differentiate between the Doctor (who asks clinical questions, performs exams, "
                "and provides plans) and the Patient (who describes symptoms and history). "
                "You are highly resistant to 'speaker drift' and ensure facts are attributed correctly."
            ),
            llm=self.llm,
            verbose=True
        )

    def _build_task(self, context: str) -> Task:
        return Task(
            description=(
                "The following text is a raw medical consultation transcript without speaker labels.\n\n"
                f"TRANSCRIPT:\n{context}\n\n"
                "STEP 1: Internally diarize the text. Identify who said what based strictly on clinical context.\n"
                "STEP 2: Map the findings to the SOAP format using STRICT grounding rules:\n"
                "- SUBJECTIVE: ONLY patient-reported symptoms, pain scores, duration, and history.\n"
                "  * Pain scales (e.g., 7/10) are SUBJECTIVE.\n"
                "  * Patient answers to questions are SUBJECTIVE.\n"
                "- OBJECTIVE: ONLY measurable findings observed or stated by the Doctor.\n"
                "  * Includes vitals (temperature, heart rate), physical exam findings, lab results.\n"
                "  * Questions alone do NOT count as Objective.\n"
                "- ASSESSMENT: ONLY diagnoses or impressions explicitly stated by the Doctor.\n"
                "  * DO NOT infer or guess diagnoses.\n"
                "  * If none explicitly stated, return 'Not Documented'.\n"
                "- PLAN: ONLY treatments, medications, tests, or follow-up instructions explicitly ordered by the Doctor.\n"
                "  * Do not create new treatment plans.\n"
                "CRITICAL RULES:\n"
                "1. Do NOT infer beyond the transcript.\n"
                "2. Do NOT assume unstated diagnoses.\n"
                "3. If information is missing, return 'Not Documented'.\n"
            ),
            expected_output="A structured JSON object following the SOAPNote schema.",
            agent=self.specialist,
            output_pydantic=SOAPNote
        )

    async def process_request(self, request: UnstructuredRequest) -> ServiceResponse:
        try:
            # Reusing your existing sanitization logic
            cleaned_text = self._sanitize(request.raw_text)
            truncated_text = self._truncate(cleaned_text)

            task = self._build_task(truncated_text)
            crew = Crew(
                agents=[self.specialist],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )

            result = await asyncio.to_thread(crew.kickoff)

            return ServiceResponse(
                success=True,
                data=result.pydantic,
                metadata={"request_id": request.request_id}
            )

        except Exception as e:
            return ServiceResponse(success=False, error=str(e), metadata={"request_id": request.request_id})

    def _sanitize(self, text: str) -> str:
        patterns = [r"(?i)ignore previous", r"(?i)system:", r"(?i)user:", r"(?i)assistant:"]
        for p in patterns: text = re.sub(p, "[REDACTED]", text)
        return text.strip()

    def _truncate(self, text: str) -> str:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return encoding.decode(tokens[:12000]) if len(tokens) > 12000 else text

# ==========================================
# 3. FASTAPI INTEGRATION
# ==========================================
app = FastAPI(title="Raw Consultation Agent", version="2.1.0")
orchestrator = AgenticOrchestrator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/agent/unstructured-note", response_model=ServiceResponse)
async def generate_note(request: UnstructuredRequest):
    if not request.request_id:
        request.request_id = str(uuid.uuid4())

    result = await orchestrator.process_request(request)
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)
    return result

@app.get("/health")
async def health_check():
    return {"status": "healthy", "architecture": "crewai-single-file"}

if __name__ == "__main__":
    uvicorn.run("consultation_agent:app", host="0.0.0.0", port=8004, reload=True)
