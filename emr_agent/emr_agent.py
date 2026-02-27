"""
EMR FastAPI Backend - Medical Document Intelligence System
Supports: PDF, Images (CT/X-ray/prescriptions), Scanned Documents
Features: OCR, AI Summarization, RAG, Patient/Doctor Chat, Name Verification
"""

import os
import sys

# Force UTF-8 output on Windows so log characters render correctly
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Redirect stderr → stdout so PowerShell never sees anything on stderr.
# PowerShell treats every stderr write as a NativeCommandError (red block),
# even for normal INFO logs. This one-line fix silences that completely.
sys.stderr = sys.stdout

import uuid
import json
import base64
import asyncio
import tempfile
import shutil
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import io

# Disable ChromaDB's anonymised telemetry BEFORE importing chromadb.
# Without this, chromadb writes an INFO message to stderr on every startup,
# which PowerShell misinterprets as a NativeCommandError.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from openai import AzureOpenAI
import chromadb
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from rapidfuzz import fuzz
import uvicorn
from dotenv import load_dotenv
load_dotenv()  # loads .env automatically when running locally

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
# Direct all log output to stdout so PowerShell does not flag it as a
# NativeCommandError (PowerShell treats anything on stderr as an error).
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("EMR-Backend")

DATA_DIR = Path("./emr_data")
DATA_DIR.mkdir(exist_ok=True)
PATIENTS_FILE  = DATA_DIR / "patients.json"
DOCS_META_FILE = DATA_DIR / "documents_meta.json"
# Persistent chroma directory — embeddings survive server restarts
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_DIR.mkdir(exist_ok=True)

# Azure OpenAI — LLM (chat / summarization)
AZURE_API_KEY           = os.getenv("AZURE_API_KEY", "")
AZURE_LLM_ENDPOINT      = os.getenv("AZURE_LLM_ENDPOINT", "")
AZURE_API_VERSION       = os.getenv("AZURE_LLM_API_VERSION", "2025-01-01-preview")
AZURE_DEPLOYMENT        = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4o-mini")

# Azure OpenAI — Embeddings (vector search / RAG)
AZURE_EMBED_ENDPOINT    = os.getenv("AZURE_EMBEDDING_ENDPOINT", "")
AZURE_EMBED_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION", "2024-02-01")
AZURE_EMBED_DEPLOYMENT  = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

DOCTOR_ACCESS_CODE  = os.getenv("DOCTOR_ACCESS_CODE", "doctor2024")
NAME_MATCH_THRESHOLD = 75  # Fuzzy match threshold %
MAX_WORKERS = 8

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

# ─────────────────────────────────────────────
# Initialize Services
# ─────────────────────────────────────────────

# LLM client (chat completions + vision)
ai_client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_LLM_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

# Embedding client (separate api_version for embeddings)
embed_client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_EMBED_ENDPOINT or AZURE_LLM_ENDPOINT,
    api_version=AZURE_EMBED_API_VERSION,
)

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


# ── Custom Azure Embedding Function for ChromaDB ──────────────────────────────
class AzureEmbeddingFunction:
    """
    Drop-in ChromaDB embedding function backed by Azure OpenAI.
    Works with PersistentClient — embeddings are stored on disk and survive restarts.

    ChromaDB 1.x dispatches three different method names depending on context:
      • __call__        — used when adding / upserting documents
      • embed_documents — alternate batch path in some 1.x builds
      • embed_query     — used when calling collection.query(query_texts=[...])
    All three delegate to the same _embed() core so there is only one code path.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    def name(self) -> str:
        # ChromaDB >= 1.x calls name() during get_or_create_collection to
        # detect conflicts with the built-in default embedding function.
        return "azure-openai-embedding"

    # ── Core embedding logic ──────────────────────────────────────────────────
    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Send texts to Azure OpenAI embeddings in batches of 64 and return vectors."""
        if not texts:
            return []
        batch_size = 64
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = embed_client.embeddings.create(
                model=AZURE_EMBED_DEPLOYMENT,
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    # ── ChromaDB protocol methods ─────────────────────────────────────────────
    def __call__(self, input: List[str]) -> List[List[float]]:
        # Called by ChromaDB when adding / upserting documents.
        return self._embed(input)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        # Alternate batch path called in some ChromaDB 1.x builds.
        return self._embed(documents)

    def embed_query(self, input: str) -> List[float]:  # noqa: A002
        # ChromaDB 1.x calls embed_query(input="...") — the kwarg MUST be named
        # `input` or it raises "unexpected keyword argument 'input'".
        # Without this working, every RAG query silently fails and the AI answers
        # from general knowledge instead of the patient's actual documents.
        result = self._embed([input])
        return result[0] if result else []


embedding_fn = AzureEmbeddingFunction()

# ── ChromaDB — persistent on disk so embeddings survive server restarts ────────
# PersistentClient writes to CHROMA_DIR and reloads automatically on next start.
# This fixes the "I don't have access to your records" bug that occurred when the
# server was restarted between document upload and chat (EphemeralClient lost all
# embeddings on every restart, so the RAG query always returned empty results).
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
chunks_collection = chroma_client.get_or_create_collection(
    name="emr_chunks",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)
summaries_collection = chroma_client.get_or_create_collection(
    name="emr_summaries",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)
logger.info(f"ChromaDB PersistentClient initialised at {CHROMA_DIR} — embeddings persist across restarts")

# ─────────────────────────────────────────────
# Data Persistence Helpers
# ─────────────────────────────────────────────
def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

def get_patients() -> dict:
    return load_json(PATIENTS_FILE)

def save_patients(data: dict):
    save_json(PATIENTS_FILE, data)

def get_docs_meta() -> dict:
    return load_json(DOCS_META_FILE)

def save_docs_meta(data: dict):
    save_json(DOCS_META_FILE, data)

# ─────────────────────────────────────────────
# AI Summarization Prompts
# Two variants:
#   SUMMARY_PROMPT_TEMPLATE  — text documents (prescriptions, lab reports, discharge notes)
#   IMAGE_SUMMARY_PROMPT     — medical images (X-ray, CT, MRI, ultrasound, etc.)
# ─────────────────────────────────────────────

# ── Shared omission rules injected into both prompts ─────────────────────────
_OMISSION_RULES = """
**ABSOLUTE OMISSION RULES — NO EXCEPTIONS**:
1. If a field has NO confirmed data, DELETE THE ENTIRE LINE — do NOT write
   "Not specified", "N/A", "Unknown", "Unavailable", or any placeholder.
2. Only include a field if you can confirm real data for it from the document.
3. Never invent or guess values.
"""

# ── Shared metadata block injected into both prompts ─────────────────────────
_NAME_EXTRACTION_BLOCK = """
**REQUIRED — NAME EXTRACTION BLOCK** (place at the very end, inside a code fence):
```
PATIENT_NAME_EXTRACTED: [Full Patient Name or NONE_FOUND]
DOCUMENT_TYPE: [Prescription/Lab Report/X-Ray/CT Scan/MRI/Ultrasound/Pathology/Medical Record/Other]
```
"""

# ── Text document prompt (prescriptions, lab reports, discharge notes) ────────
SUMMARY_PROMPT_TEMPLATE = """You are a senior radiologist/physician AI. Analyze the following medical document and produce a detailed, clinically accurate structured summary in Markdown.
{omission_rules}

**DOCTOR NAME EXTRACTION**: Look carefully for names near "Dr.", "MD", "MBBS", signature lines, letterheads, stamps, or seals. If handwritten, attempt to decipher.

**INSTRUCTIONS**:
1. Start with available patient/doctor/hospital info (skip any field with no confirmed data — no placeholders).
2. List every medicine on its own bullet with dosage, frequency, and a sub-bullet explaining its purpose.
3. If lab results or vitals are present, add a Markdown table at the bottom.
4. Write clinical insights in plain English so a non-medical person can understand.

**OUTPUT STRUCTURE**:

### 🏥 Hospital & Patient Profile
*(Only include lines where you have confirmed data)*
- **Hospital/Clinic**: [Name]
- **Doctor Name**: [Full Name with Title]
- **Patient Name**: [Name]
- **Age/Sex**: [Age] / [Sex]
- **Date**: [Date]

### 🩺 Comprehensive Medical Summary
- **Key Findings / Complaints**: [Decoded & simplified from the document]
- **Diagnosis / Impression**: [Doctor's conclusion]
- **Clinical Insights**: [Plain-English explanation of what this means for the patient]

### 💊 Treatment & Medications
- **[Medicine Name]** — [Dosage], [Frequency]
  - *Purpose*: [Why this medicine is prescribed]
- **Lifestyle / Advice**: [Diet, rest, activity restrictions, follow-up date]

### 📊 Lab Results / Vitals
*(Markdown table — only if data exists; skip this section entirely if not)*
| Parameter | Value | Reference Range | Status |
|-----------|-------|----------------|--------|

### ⚠️ Medical Disclaimer
*This AI-generated summary may contain errors, especially from handwritten text. It is not a substitute for professional medical advice — verify all details with your doctor.*

---
**Document to Analyze:**
{content_data}
{name_extraction_block}"""

# ── Medical image prompt (X-ray, CT, MRI, ultrasound, nuclear medicine) ──────
IMAGE_SUMMARY_PROMPT = """You are a senior radiologist AI with expert-level visual interpretation skills. A medical imaging file has been provided. Perform a thorough, structured radiological analysis as a real radiologist would write in an official report.
{omission_rules}

**IMAGING ANALYSIS RULES**:
1. Identify the modality (X-ray/CT/MRI/Ultrasound/PET etc.) and body part/region.
2. Describe the view/plane (AP, lateral, coronal, axial, sagittal, etc.) if visible.
3. Systematically examine EVERY visible structure — bones, joints, soft tissues, organs, vasculature, airways, foreign bodies.
4. For EACH finding describe: location, size/extent, severity, and clinical significance.
5. State your radiological impression clearly with a differential diagnosis ranked by likelihood.
6. Give specific, actionable clinical recommendations.
7. If it is an X-ray, specifically comment on: bone density, cortical integrity, joint spaces, alignment, and soft tissue shadows.
8. If it is a CT/MRI, comment on: attenuation/signal characteristics, margins, enhancement pattern, mass effect, surrounding structure involvement.
9. Do NOT write placeholder phrases like "Not specified" — skip any field you cannot visually confirm.

**OUTPUT STRUCTURE**:

### 🏥 Patient & Scan Profile
*(Only include lines you can confirm from the image or its metadata)*
- **Patient Name**: [If visible on image header]
- **Age/Sex**: [If visible]
- **Date of Scan**: [If visible]
- **Referring Doctor**: [If visible]
- **Institution**: [If visible]

### 🔬 Imaging Details
- **Modality**: [X-ray / CT / MRI / Ultrasound / Other]
- **Body Region**: [e.g. Left Knee, Chest, Abdomen, Brain]
- **View / Plane**: [e.g. AP + Lateral, Axial, Coronal]
- **Image Quality**: [Adequate / Suboptimal — note any limiting factors]

### 🦴 Systematic Findings
*(Examine every visible structure methodically)*

**Bones & Joints**:
- [Detailed finding with location, description, severity]
- [Continue for each structure…]

**Soft Tissues**:
- [Swelling, effusion, calcification, foreign body, etc.]

**Other Structures**:
- [Any additional visible structures — organs, vessels, airways as relevant]

### 🩻 Radiological Impression
**Primary Diagnosis**: [Most likely diagnosis with reasoning]
**Differential Diagnoses**:
1. [Most likely alternative]
2. [Second alternative if applicable]

**Severity Assessment**: [Mild / Moderate / Severe / Critical] — [one-line justification]

### 💡 Clinical Recommendations
- **Immediate Action**: [e.g. Orthopaedic referral, surgical consultation, conservative management]
- **Further Imaging**: [e.g. MRI for soft tissue detail, weight-bearing X-ray, CT for fracture characterization]
- **Follow-up**: [Timeline and what to monitor]
- **Red Flags to Watch**: [Symptoms that would require urgent re-evaluation]

### 📋 Report Summary
*(2–3 sentence plain-English summary for the patient / non-specialist)*
[Write this so a patient without medical training can understand what was found, how serious it is, and what happens next.]

### ⚠️ Medical Disclaimer
*This AI-generated radiological interpretation is for informational purposes only. It must be reviewed and confirmed by a qualified radiologist or physician before any clinical decision is made.*

---
**Image to Analyze:** {content_data}
{name_extraction_block}"""

SYSTEM_PROMPT = "You are a highly skilled medical AI combining the expertise of a senior physician, radiologist, and clinical pharmacist. Output using detailed, clinically precise Markdown with relevant emojis."

# ─────────────────────────────────────────────
# Document Processing
# ─────────────────────────────────────────────
def extract_text_from_pdf(file_path: str) -> tuple[str, bool]:
    """Returns (text, used_ocr)"""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        if len(text.strip()) > 50:
            logger.info(f"PDF text extracted directly: {len(text)} chars")
            return text, False
        else:
            logger.info("PDF has minimal text, converting to images for OCR")
            return extract_text_via_ocr_from_pdf(file_path), True
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return "", False

def extract_text_via_ocr_from_pdf(file_path: str) -> str:
    """Convert PDF pages to images and OCR them"""
    try:
        images = convert_from_path(file_path, dpi=300)
        all_text = []
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, config='--psm 6')
            all_text.append(f"[Page {i+1}]\n{text}")
        return "\n\n".join(all_text)
    except Exception as e:
        logger.error(f"OCR from PDF error: {e}")
        return ""

def extract_text_from_image_ocr(file_path: str) -> str:
    """OCR from image file"""
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image, config='--psm 6')
        return text
    except Exception as e:
        logger.error(f"Image OCR error: {e}")
        return ""

def image_to_base64(file_path: str) -> tuple[str, str]:
    """Convert image to base64 data URL for Azure OpenAI vision"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.standard_b64encode(data).decode("utf-8")
    
    ext = Path(file_path).suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".bmp": "image/bmp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")
    return b64, media_type

async def process_document_with_ai(file_path: str, filename: str, use_vision: bool = False, ocr_text: str = "") -> str:
    """Send document to Azure OpenAI for summarization.

    Uses IMAGE_SUMMARY_PROMPT for medical images (X-ray, CT, MRI, ultrasound, etc.)
    and SUMMARY_PROMPT_TEMPLATE for all text-based documents.
    Both prompts share the same strict omission rules and name-extraction footer.
    """
    loop = asyncio.get_event_loop()

    def _call_azure():
        # Shared keyword arguments injected into both prompt templates
        shared_kwargs = {
            "omission_rules": _OMISSION_RULES,
            "name_extraction_block": _NAME_EXTRACTION_BLOCK,
        }

        if use_vision:
            # ── Medical image path — use dedicated radiology prompt ───────────
            b64, media_type = image_to_base64(file_path)
            data_url = f"data:{media_type};base64,{b64}"

            prompt_text = IMAGE_SUMMARY_PROMPT.format(
                content_data=f"[{filename}] — analyze the image provided above.",
                **shared_kwargs,
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        # Send the image first so the model sees it before the instructions
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]
        else:
            # ── Text document path — standard clinical summary prompt ─────────
            content_data = ocr_text if ocr_text else f"[No text could be extracted from {filename}]"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": SUMMARY_PROMPT_TEMPLATE.format(
                        content_data=content_data,
                        **shared_kwargs,
                    ),
                },
            ]

        response = ai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=messages,
            max_tokens=4096,
            temperature=0.1,   # Lower temperature → more factual, less hallucination
        )
        return response.choices[0].message.content

    return await loop.run_in_executor(executor, _call_azure)

def extract_patient_name_from_summary(summary: str) -> Optional[str]:
    """Extract the PATIENT_NAME_EXTRACTED from AI summary"""
    for line in summary.split('\n'):
        if 'PATIENT_NAME_EXTRACTED:' in line:
            name = line.split('PATIENT_NAME_EXTRACTED:')[-1].strip()
            if name and name.upper() != 'NONE_FOUND' and len(name) > 1:
                return name
    return None

def extract_doc_type_from_summary(summary: str) -> str:
    """Extract document type from AI summary"""
    for line in summary.split('\n'):
        if 'DOCUMENT_TYPE:' in line:
            return line.split('DOCUMENT_TYPE:')[-1].strip()
    return "Medical Document"

def check_name_match(registered_name: str, extracted_name: Optional[str]) -> tuple[bool, int]:
    """
    Smart multi-strategy name matcher — returns (is_match, best_score).

    Strategies tried (highest score wins):
    1. token_set_ratio  — handles partial names, e.g. "Surya" matches
                          "Surya Pranav Perumalla" because all tokens of the
                          shorter string are present in the longer one → 100.
    2. partial_ratio    — substring match, e.g. "Surya" is fully contained
                          inside "Surya Pranav Perumalla" → 100.
    3. token_sort_ratio — classic sorted-token match, good for reordered names.
    4. Token-subset check — explicit check: every word in the registered name
                            appears as a word in the extracted name → 100.

    Using the max of all four means a nickname / first-name-only registration
    will never produce a false mismatch warning.
    """
    if not extracted_name:
        return True, 0  # Benefit of the doubt when no name is found in doc

    reg  = registered_name.lower().strip()
    ext  = extracted_name.lower().strip()

    # Strategy 1 — token set ratio (best for partial / subset names)
    s1 = fuzz.token_set_ratio(reg, ext)

    # Strategy 2 — partial ratio (substring containment)
    s2 = fuzz.partial_ratio(reg, ext)

    # Strategy 3 — token sort ratio (classic)
    s3 = fuzz.token_sort_ratio(reg, ext)

    # Strategy 4 — explicit word-subset check
    reg_words = set(reg.split())
    ext_words = set(ext.split())
    s4 = 100 if reg_words and reg_words.issubset(ext_words) else 0

    best = max(s1, s2, s3, s4)
    return best >= NAME_MATCH_THRESHOLD, best

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for vector storage"""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks if chunks else [text]

async def store_in_vector_db(patient_id: str, doc_id: str, filename: str, 
                              summary: str, raw_text: str, doc_type: str):
    """Store document chunks and summary in ChromaDB"""
    loop = asyncio.get_event_loop()
    
    def _store():
        # Store summary
        summaries_collection.upsert(
            ids=[doc_id],
            documents=[summary],
            metadatas=[{
                "patient_id": patient_id,
                "doc_id": doc_id,
                "filename": filename,
                "doc_type": doc_type,
                "timestamp": datetime.now().isoformat()
            }]
        )
        
        # Store raw text chunks
        text_to_chunk = raw_text if raw_text else summary
        chunks = chunk_text(text_to_chunk)
        
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "patient_id": patient_id,
                "doc_id": doc_id,
                "filename": filename,
                "doc_type": doc_type,
                "chunk_index": i,
                "timestamp": datetime.now().isoformat()
            }
            for i in range(len(chunks))
        ]
        
        chunks_collection.upsert(
            ids=chunk_ids,
            documents=chunks,
            metadatas=metadatas
        )
        logger.info(f"Stored {len(chunks)} chunks for doc {doc_id}")
    
    await loop.run_in_executor(executor, _store)

# ─────────────────────────────────────────────
# RAG Query
# ─────────────────────────────────────────────
def query_patient_documents(patient_id: str, query: str, n_results: int = 5) -> List[Dict]:
    """
    Query vector DB for patient-specific document chunks.

    We compute the query embedding ourselves via embedding_fn._embed() and pass
    query_embeddings= to ChromaDB instead of query_texts=.  This bypasses
    ChromaDB's internal embedding-dispatch (embed_query / __call__) which has
    inconsistent keyword-argument conventions across 1.x patch versions and was
    causing '$.input' is invalid 400 errors from the Azure OpenAI API.
    """
    try:
        # Step 1 — embed the query string directly through our Azure client
        query_vec = embedding_fn._embed([query])
        if not query_vec:
            logger.warning("query_patient_documents: embedding returned empty — skipping vector search")
            return []

        # Step 2 — query ChromaDB with the pre-computed vector
        results = chunks_collection.query(
            query_embeddings=[query_vec[0]],   # pre-computed; no embed_query call
            n_results=n_results,
            where={"patient_id": patient_id}
        )

        contexts = []
        if results and results['documents']:
            # Load docs_meta once so we can enrich every chunk with name-verification status.
            # This lets generate_chat_response warn the AI (and therefore the user) whenever
            # it is answering from a document whose patient name could not be confirmed.
            docs_meta = get_docs_meta()

            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i] if results['metadatas'] else {}
                doc_id = meta.get("doc_id", "")

                # Look up name-match status from the persisted document metadata
                stored = docs_meta.get(doc_id, {})
                name_match    = stored.get("name_match")      # True / False / None
                extracted_name = stored.get("extracted_name") # str or None

                contexts.append({
                    "text": doc,
                    "filename": meta.get("filename", "Unknown"),
                    "doc_type": meta.get("doc_type", "Unknown"),
                    "distance": results['distances'][0][i] if results.get('distances') else 0,
                    # Verification fields forwarded to the chat response builder
                    "name_match": name_match,
                    "extracted_name": extracted_name,
                })
        return contexts
    except Exception as e:
        logger.error(f"Vector query error: {e}")
        return []

async def generate_chat_response(
    patient_id: str,
    message: str,
    conversation_history: List[Dict],
    is_doctor: bool = False,
    patient_name: str = "",
    doctor_name: str = "",
) -> str:
    """Generate RAG-based chat response"""
    # Retrieve relevant context
    contexts = query_patient_documents(patient_id, message, n_results=6)

    context_text = ""
    if contexts:
        context_text = "\n\n**Relevant Medical Records:**\n"
        for ctx in contexts:
            name_match     = ctx.get("name_match")       # True / False / None
            extracted_name = ctx.get("extracted_name")   # str or None

            # Build a human-readable verification tag that travels with the chunk
            # so the AI knows exactly how trustworthy this piece of data is.
            if name_match is True:
                # Name in document matches the registered patient — no warning needed
                verification_tag = f"[Name verified: {extracted_name}]"
            elif name_match is False:
                # Document contains a different name — could be a wrong upload
                verification_tag = (
                    f"[⚠️ NAME MISMATCH — document contains '{extracted_name}', "
                    f"but this patient is registered as '{patient_name}'. "
                    f"Data may not belong to this patient.]"
                )
            else:
                # No name was detected in the document at all
                verification_tag = "[⚠️ NAME NOT DETECTED — patient identity in this document could not be confirmed.]"

            context_text += (
                f"\n📄 [{ctx['filename']} - {ctx['doc_type']}] {verification_tag}:\n"
                f"{ctx['text']}\n"
            )

    if is_doctor:
        # Address the doctor by name when available so the AI greets them personally
        doctor_address = f"Dr. {doctor_name}" if doctor_name.strip() else "Doctor"
        role_context = (
            f"You are a clinical AI assistant speaking with **{doctor_address}** about patient **{patient_name}**. "
            f"Address the doctor as '{doctor_address}' when appropriate (e.g. greeting, clarifications). "
            f"Provide clinical, professional analysis. Be precise with medical terminology."
        )
    else:
        role_context = (
            f"You are a friendly medical AI assistant speaking with **{patient_name}** (the patient). "
            f"Explain things in simple, easy-to-understand language. Be empathetic and supportive."
        )

    system = f"""{role_context}

You have access to the patient's medical records. Use the provided context to give accurate, personalized answers.
Always cite which document you're referencing. If information is not in the records, say so clearly.
Never make up medical information. Recommend consulting a doctor for treatment decisions.

IMPORTANT — Name verification rules (follow strictly):
- Each document in the context has a verification tag: [Name verified], [⚠️ NAME MISMATCH], or [⚠️ NAME NOT DETECTED].
- If you use data from a document tagged NAME MISMATCH or NAME NOT DETECTED, you MUST mention it clearly in your answer.
  Example: "Based on the report '{{filename}}', however please note the name in that document does not match your registered name, so please verify this is your document."
- Never silently use data from an unverified document without flagging it to the user.
- If ALL retrieved documents are unverified, lead your answer with a clear caution before providing the data.
"""
    
    # Build message list
    messages = []
    for h in conversation_history[-10:]:  # Keep last 10 turns
        messages.append({"role": h["role"], "content": h["content"]})
    
    # Add current message with context
    user_message = f"{message}\n\n{context_text}" if context_text else message
    messages.append({"role": "user", "content": user_message})
    
    loop = asyncio.get_event_loop()
    
    def _call():
        az_messages = [{"role": "system", "content": system}] + messages
        response = ai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=az_messages,
            max_tokens=2048,
            temperature=0.3,
        )
        return response.choices[0].message.content
    
    return await loop.run_in_executor(executor, _call)

# ─────────────────────────────────────────────
# Startup Re-indexer
# ─────────────────────────────────────────────
def reindex_from_disk():
    """
    Re-populate ChromaDB from the saved documents_meta.json when the collections
    are empty (e.g. chroma dir was wiped while docs_meta.json still has data).
    This ensures the RAG chat works immediately after any kind of server restart,
    even if the persisted chroma files were lost.

    We index both the AI summary and the raw chunks stored in the metadata.
    Because raw_text is not persisted to disk (only summary + length are),
    we fall back to the summary text for chunking — the summary contains all
    medically relevant information extracted by the AI, so RAG still works well.
    """
    try:
        # Only run if the chunks collection is empty but we have saved metadata
        if chunks_collection.count() > 0:
            logger.info(f"ChromaDB already has {chunks_collection.count()} chunks — skipping re-index")
            return

        docs_meta = load_json(DOCS_META_FILE)
        if not docs_meta:
            logger.info("No documents metadata found — nothing to re-index")
            return

        logger.info(f"ChromaDB is empty but {len(docs_meta)} documents exist on disk — re-indexing…")

        reindexed = 0
        for doc_id, doc in docs_meta.items():
            try:
                summary   = doc.get("summary", "")
                patient_id = doc.get("patient_id", "")
                filename   = doc.get("filename", "")
                doc_type   = doc.get("doc_type", "Medical Document")
                timestamp  = doc.get("timestamp", "")

                if not summary or not patient_id:
                    continue

                # Re-index summary into summaries_collection
                summaries_collection.upsert(
                    ids=[doc_id],
                    documents=[summary],
                    metadatas=[{
                        "patient_id": patient_id,
                        "doc_id":     doc_id,
                        "filename":   filename,
                        "doc_type":   doc_type,
                        "timestamp":  timestamp,
                    }]
                )

                # Re-chunk the summary and index into chunks_collection
                # (raw text is not persisted — summary is the best available fallback)
                chunks    = chunk_text(summary)
                chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
                metadatas = [
                    {
                        "patient_id":  patient_id,
                        "doc_id":      doc_id,
                        "filename":    filename,
                        "doc_type":    doc_type,
                        "chunk_index": i,
                        "timestamp":   timestamp,
                    }
                    for i in range(len(chunks))
                ]
                chunks_collection.upsert(
                    ids=chunk_ids,
                    documents=chunks,
                    metadatas=metadatas,
                )
                reindexed += 1

            except Exception as e:
                logger.warning(f"Re-index failed for doc {doc_id}: {e}")

        logger.info(f"Re-index complete — {reindexed}/{len(docs_meta)} documents restored into ChromaDB")

    except Exception as e:
        logger.error(f"Re-index error: {e}")


# Run re-indexer synchronously at import time (before the first request arrives)
reindex_from_disk()


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="EMR Intelligence System",
    description="AI-powered Electronic Medical Record System with RAG",
    version="1.0.0"
)

ALLOWED_ORIGINS = [
    "https://healthcareagents.dimensionleap.com",
    "https://dimensionleap-ai-health.vercel.app",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ─────────────────────────────────────────────
# Auto-purge — inactive patient cleanup
# ─────────────────────────────────────────────
# Any patient who has had NO activity for INACTIVITY_TIMEOUT_SECONDS is
# automatically deleted, together with:
#   • their entry in patients.json
#   • all their documents in documents_meta.json
#   • all their ChromaDB embeddings (both chunks and summary collections)
#
# A background asyncio task (run_auto_purge) fires every
# PURGE_CHECK_INTERVAL_SECONDS and calls purge_inactive_patients().

INACTIVITY_TIMEOUT_SECONDS = 30 * 60          # 30 minutes
PURGE_CHECK_INTERVAL_SECONDS = 5 * 60         # check every 5 minutes


def _touch_patient(patient_id: str) -> None:
    """Stamp the current UTC time as last_activity for the given patient.
    Called from every API route that involves a specific patient so the
    30-minute inactivity clock is reset on each interaction."""
    patients = get_patients()
    if patient_id in patients:
        patients[patient_id]["last_activity"] = datetime.utcnow().isoformat()
        save_patients(patients)


def purge_inactive_patients() -> None:
    """Delete all patients (+ their documents + ChromaDB vectors) that have
    not been active for at least INACTIVITY_TIMEOUT_SECONDS.

    Deletion is thorough:
      1. Remove the patient from patients.json
      2. Remove every document owned by the patient from documents_meta.json
      3. Delete all ChromaDB entries (chunks + summaries) whose metadata
         field patient_id matches — this frees the embedding storage.
    """
    now = datetime.utcnow()
    patients = get_patients()
    docs_meta = get_docs_meta()

    # Collect patient IDs that have exceeded the inactivity window
    expired: List[str] = []
    for pid, pdata in patients.items():
        raw_ts = pdata.get("last_activity") or pdata.get("registered_at", "")
        if not raw_ts:
            # No timestamp at all — treat as expired to be safe
            expired.append(pid)
            continue
        try:
            last_seen = datetime.fromisoformat(raw_ts)
        except ValueError:
            expired.append(pid)
            continue
        if (now - last_seen).total_seconds() >= INACTIVITY_TIMEOUT_SECONDS:
            expired.append(pid)

    if not expired:
        return  # Nothing to do

    logger.info(f"Auto-purge: removing {len(expired)} inactive patient(s): "
                f"{[patients[p]['name'] for p in expired]}")

    # ── 1. Remove ChromaDB entries for each expired patient ──────────────────
    for pid in expired:
        # Find all doc_ids that belong to this patient (for summaries collection)
        patient_doc_ids = [
            doc_id for doc_id, d in docs_meta.items() if d.get("patient_id") == pid
        ]
        # Delete summaries (keyed by doc_id)
        if patient_doc_ids:
            try:
                summaries_collection.delete(ids=patient_doc_ids)
            except Exception as e:
                logger.warning(f"Purge: could not delete summaries for {pid}: {e}")

        # Delete chunks (keyed by patient_id metadata field)
        try:
            chunks_collection.delete(where={"patient_id": pid})
        except Exception as e:
            logger.warning(f"Purge: could not delete chunks for {pid}: {e}")

    # ── 2. Remove documents from docs_meta ───────────────────────────────────
    docs_meta = {
        doc_id: d
        for doc_id, d in docs_meta.items()
        if d.get("patient_id") not in expired
    }
    save_docs_meta(docs_meta)

    # ── 3. Remove patients from patients.json ────────────────────────────────
    for pid in expired:
        patients.pop(pid, None)
    save_patients(patients)

    logger.info("Auto-purge complete.")


async def run_auto_purge() -> None:
    """Background coroutine — runs for the lifetime of the server.
    Sleeps for PURGE_CHECK_INTERVAL_SECONDS between each purge sweep."""
    # Wait one full interval before the first check so the server has time to
    # finish startup and re-indexing before we start deleting stale records.
    await asyncio.sleep(PURGE_CHECK_INTERVAL_SECONDS)
    while True:
        try:
            # Run the blocking file-and-DB work in a thread so we don't stall
            # the event loop during the JSON reads/writes and ChromaDB deletes.
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, purge_inactive_patients)
        except Exception as e:
            logger.error(f"Auto-purge sweep failed: {e}")
        await asyncio.sleep(PURGE_CHECK_INTERVAL_SECONDS)


@app.on_event("startup")
async def start_background_tasks() -> None:
    """Kick off the auto-purge background task when the server starts."""
    asyncio.create_task(run_auto_purge())
    logger.info(
        f"Auto-purge started — patients inactive for "
        f"{INACTIVITY_TIMEOUT_SECONDS // 60} min will be deleted "
        f"(check interval: {PURGE_CHECK_INTERVAL_SECONDS // 60} min)"
    )

# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────
class PatientRegister(BaseModel):
    name: str

class ChatRequest(BaseModel):
    patient_id: str
    message: str
    conversation_history: List[Dict] = []

class DoctorChatRequest(BaseModel):
    patient_id: str
    message: str
    conversation_history: List[Dict] = []
    # Optional — when provided the AI greets and addresses the doctor by name
    doctor_name: str = ""

class PatientLookup(BaseModel):
    name: str

# ─────────────────────────────────────────────
# Routes - Patient Management
# ─────────────────────────────────────────────
@app.post("/api/patients/register")
async def register_patient(data: PatientRegister):
    """Register a new patient or return existing one"""
    patients = get_patients()
    
    # Check if patient already exists using the same multi-strategy matcher.
    # Re-login threshold is intentionally high (90) to avoid false merges,
    # but token_set_ratio means "Surya" still re-logs into "Surya Pranav Perumalla".
    for pid, pdata in patients.items():
        reg = data.name.lower().strip()
        ext = pdata["name"].lower().strip()
        score = max(
            fuzz.token_set_ratio(reg, ext),
            fuzz.partial_ratio(reg, ext),
            fuzz.token_sort_ratio(reg, ext),
            100 if set(reg.split()).issubset(set(ext.split())) else 0,
        )
        if score >= 90:  # High threshold to avoid merging different patients
            logger.info(f"Patient re-login: {pdata['name']}")
            # Re-login counts as activity — reset the inactivity clock
            _touch_patient(pid)
            return {"patient_id": pid, "name": pdata["name"], "is_new": False}

    # Create new patient — seed last_activity so the purge clock starts now
    patient_id = str(uuid.uuid4())
    patients[patient_id] = {
        "name": data.name.strip(),
        "registered_at": datetime.utcnow().isoformat(),
        "last_activity": datetime.utcnow().isoformat(),
        "doc_count": 0
    }
    save_patients(patients)
    logger.info(f"New patient registered: {data.name} -> {patient_id}")
    return {"patient_id": patient_id, "name": data.name.strip(), "is_new": True}

@app.get("/api/patients/list")
async def list_all_patients():
    """Return every registered patient with their name, id, and document count.
    Declared BEFORE the /{patient_id} dynamic routes so FastAPI matches it first.
    Used by the doctor portal dropdown so doctors can pick a patient without typing."""
    patients = get_patients()
    docs_meta = get_docs_meta()
    result = []
    for pid, pdata in patients.items():
        doc_count = len([d for d in docs_meta.values() if d.get("patient_id") == pid])
        result.append({
            "patient_id": pid,
            "name": pdata["name"],
            "doc_count": doc_count,
            "registered_at": pdata.get("registered_at", ""),
        })
    # Sort alphabetically by name so the dropdown is easy to scan
    result.sort(key=lambda x: x["name"].lower())
    return {"patients": result}

@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str):
    patients = get_patients()
    if patient_id not in patients:
        raise HTTPException(404, "Patient not found")
    return patients[patient_id]

@app.get("/api/patients/{patient_id}/has-documents")
async def patient_has_documents(patient_id: str):
    docs_meta = get_docs_meta()
    patient_docs = [d for d in docs_meta.values() if d.get("patient_id") == patient_id]
    return {"has_documents": len(patient_docs) > 0, "count": len(patient_docs)}

@app.post("/api/patients/lookup")
async def lookup_patient_by_name(data: PatientLookup):
    """Find a patient by name (for doctor access)"""
    patients = get_patients()
    matches = []
    for pid, pdata in patients.items():
        reg = data.name.lower().strip()
        ext = pdata["name"].lower().strip()
        score = max(
            fuzz.token_set_ratio(reg, ext),
            fuzz.partial_ratio(reg, ext),
            fuzz.token_sort_ratio(reg, ext),
            100 if set(reg.split()).issubset(set(ext.split())) else 0,
        )
        if score >= 70:
            docs_meta = get_docs_meta()
            doc_count = len([d for d in docs_meta.values() if d.get("patient_id") == pid])
            matches.append({
                "patient_id": pid,
                "name": pdata["name"],
                "match_score": score,
                "doc_count": doc_count,
                "registered_at": pdata.get("registered_at", "")
            })
    matches.sort(key=lambda x: x["match_score"], reverse=True)
    return {"matches": matches}

# ─────────────────────────────────────────────
# Routes - Document Upload & Processing
# ─────────────────────────────────────────────
@app.post("/api/documents/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    patient_id: str = Form(...),
    registered_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Upload and process multiple medical documents"""
    patients = get_patients()
    if patient_id not in patients:
        raise HTTPException(404, "Patient not found")

    # Uploading a document is patient activity — reset the inactivity clock
    _touch_patient(patient_id)

    results = []
    docs_meta = get_docs_meta()
    
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"Unsupported file type: {ext}"
            })
            continue
        
        doc_id = str(uuid.uuid4())
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            use_vision = False
            ocr_text = ""
            raw_text = ""
            
            is_image = ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            
            if ext == ".pdf":
                raw_text, used_ocr = extract_text_from_pdf(tmp_path)
                ocr_text = raw_text
                use_vision = False
            elif is_image:
                # For medical images (CT, X-ray), use Claude vision
                # Also try OCR in parallel for text-based images (prescriptions)
                ocr_text = extract_text_from_image_ocr(tmp_path)
                use_vision = True  # Always use vision for images
                raw_text = ocr_text
            else:
                # TIFF and other formats - try OCR
                ocr_text = extract_text_from_image_ocr(tmp_path)
                raw_text = ocr_text
                use_vision = False
            
            # Get AI summary
            summary = await process_document_with_ai(
                tmp_path, file.filename,
                use_vision=use_vision,
                ocr_text=ocr_text
            )
            
            # Extract metadata from summary
            extracted_name = extract_patient_name_from_summary(summary)
            doc_type = extract_doc_type_from_summary(summary)
            name_match, match_score = check_name_match(registered_name, extracted_name)
            
            # Store in vector DB
            await store_in_vector_db(
                patient_id, doc_id, file.filename,
                summary, raw_text, doc_type
            )
            
            # Save metadata
            docs_meta[doc_id] = {
                "doc_id": doc_id,
                "patient_id": patient_id,
                "filename": file.filename,
                "doc_type": doc_type,
                "summary": summary,
                "extracted_name": extracted_name,
                "name_match": name_match,
                "match_score": match_score,
                "registered_name": registered_name,
                "timestamp": datetime.now().isoformat(),
                "file_ext": ext,
                "used_vision": use_vision,
                "raw_text_length": len(raw_text)
            }
            
            results.append({
                "doc_id": doc_id,
                "filename": file.filename,
                "status": "success",
                "doc_type": doc_type,
                "extracted_name": extracted_name,
                "name_match": name_match,
                "match_score": match_score,
                "message": "Document processed and indexed successfully"
            })
            
            logger.info(f"Processed: {file.filename} | Match: {name_match} ({match_score}%)")
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
        finally:
            os.unlink(tmp_path)
    
    # Update patient doc count
    if patient_id in patients:
        patients[patient_id]["doc_count"] = len([
            d for d in docs_meta.values() if d.get("patient_id") == patient_id
        ])
        save_patients(patients)
    
    save_docs_meta(docs_meta)
    return {"results": results, "total": len(results)}

@app.get("/api/documents/{patient_id}")
async def list_patient_documents(patient_id: str):
    """List all documents for a patient"""
    # Viewing documents is activity — reset the inactivity clock
    _touch_patient(patient_id)
    docs_meta = get_docs_meta()
    patient_docs = [
        {
            "doc_id": d["doc_id"],
            "filename": d["filename"],
            "doc_type": d.get("doc_type", "Unknown"),
            "extracted_name": d.get("extracted_name"),
            "name_match": d.get("name_match", True),
            "match_score": d.get("match_score", 0),
            "timestamp": d.get("timestamp"),
            "has_summary": bool(d.get("summary"))
        }
        for d in docs_meta.values()
        if d.get("patient_id") == patient_id
    ]
    patient_docs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return {"documents": patient_docs}

@app.get("/api/documents/{patient_id}/{doc_id}/summary")
async def get_document_summary(patient_id: str, doc_id: str):
    """Get AI summary for a specific document"""
    # Viewing a summary is activity — reset the inactivity clock
    _touch_patient(patient_id)
    docs_meta = get_docs_meta()

    if doc_id not in docs_meta:
        raise HTTPException(404, "Document not found")

    doc = docs_meta[doc_id]
    if doc.get("patient_id") != patient_id:
        raise HTTPException(403, "Access denied")
    
    return {
        "doc_id": doc_id,
        "filename": doc["filename"],
        "summary": doc.get("summary", "Summary not available"),
        "doc_type": doc.get("doc_type", "Unknown"),
        "extracted_name": doc.get("extracted_name"),
        "name_match": doc.get("name_match", True),
        "match_score": doc.get("match_score", 0),
        "registered_name": doc.get("registered_name"),
        "timestamp": doc.get("timestamp")
    }

# ─────────────────────────────────────────────
# Routes - Chat
# ─────────────────────────────────────────────
@app.post("/api/chat/patient")
async def patient_chat(request: ChatRequest):
    """Patient chat - RAG over their own documents"""
    patients = get_patients()
    if request.patient_id not in patients:
        raise HTTPException(404, "Patient not found")

    # Chatting is activity — reset the inactivity clock
    _touch_patient(request.patient_id)

    # Check if patient has documents
    docs_meta = get_docs_meta()
    patient_docs = [d for d in docs_meta.values() if d.get("patient_id") == request.patient_id]
    if not patient_docs:
        raise HTTPException(400, "No documents uploaded yet. Please upload your medical documents first.")

    patient_name = patients[request.patient_id]["name"]

    response = await generate_chat_response(
        patient_id=request.patient_id,
        message=request.message,
        conversation_history=request.conversation_history,
        is_doctor=False,
        patient_name=patient_name
    )

    return {"response": response, "patient_name": patient_name}

@app.post("/api/chat/doctor")
async def doctor_chat(request: DoctorChatRequest):
    """Doctor chat - access patient records"""
    patients = get_patients()
    if request.patient_id not in patients:
        raise HTTPException(404, "Patient not found")

    # Doctor querying a patient's records counts as that patient's activity
    _touch_patient(request.patient_id)

    # Check if patient has documents
    docs_meta = get_docs_meta()
    patient_docs = [d for d in docs_meta.values() if d.get("patient_id") == request.patient_id]
    if not patient_docs:
        raise HTTPException(400, "Patient has not uploaded any documents yet.")

    patient_name = patients[request.patient_id]["name"]
    
    response = await generate_chat_response(
        patient_id=request.patient_id,
        message=request.message,
        conversation_history=request.conversation_history,
        is_doctor=True,
        patient_name=patient_name,
        doctor_name=request.doctor_name,
    )

    return {"response": response, "patient_name": patient_name}

# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "embedding_backend": f"Azure OpenAI / {AZURE_EMBED_DEPLOYMENT}",
        "vector_store": "ChromaDB EphemeralClient (session-only)",
        "chroma_chunks": chunks_collection.count(),
        "chroma_summaries": summaries_collection.count()
    }

@app.get("/")
async def root():
    return {"message": "EMR Intelligence System API", "version": "1.0.0", "docs": "/docs"}

# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import platform
    is_windows = platform.system() == "Windows"

    if is_windows:
        # Windows does NOT support forked multiprocessing with uvicorn.
        # Concurrency is handled by:
        #   1. FastAPI async event loop (handles many simultaneous requests)
        #   2. ThreadPoolExecutor(MAX_WORKERS=8) for CPU-bound OCR and AI calls
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8010,
            log_level="info",
        )
    else:
        # Linux / macOS: true multi-process workers
        uvicorn.run(
            "backend:app",
            host="0.0.0.0",
            port=8010,
            workers=4,
            log_level="info",
        )