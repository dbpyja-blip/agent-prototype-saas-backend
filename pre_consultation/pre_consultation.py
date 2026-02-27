from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai import LLM
import json

# Load env vars
load_dotenv()
load_dotenv(".env.local", override=True)


# ============================================================================
# AZURE OPENAI SETUP
# ============================================================================
def get_azure_llm_for_crewai(model_name: Optional[str] = None):
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_LLM_ENDPOINT = (
        os.getenv("AZURE_LLM_ENDPOINT")
        or os.getenv("AZURE_ENDPOINT")
        or os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    if AZURE_LLM_ENDPOINT:
        AZURE_LLM_ENDPOINT = AZURE_LLM_ENDPOINT.rstrip("/")
    AZURE_LLM_API_VERSION = "2024-02-01"
    AZURE_LLM_DEPLOYMENT = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4o-mini")
    deployment = model_name or AZURE_LLM_DEPLOYMENT

    if not AZURE_API_KEY:
        raise ValueError("AZURE_API_KEY not set")
    if not AZURE_LLM_ENDPOINT:
        raise ValueError("AZURE_LLM_ENDPOINT not set")

    for k, v in {
        "AZURE_API_KEY": AZURE_API_KEY,
        "AZURE_API_BASE": AZURE_LLM_ENDPOINT,
        "AZURE_ENDPOINT": AZURE_LLM_ENDPOINT,
        "AZURE_API_VERSION": AZURE_LLM_API_VERSION,
        "AZURE_OPENAI_ENDPOINT": AZURE_LLM_ENDPOINT,
        "AZURE_OPENAI_API_KEY": AZURE_API_KEY,
        "OPENAI_API_TYPE": "azure",
        "OPENAI_API_BASE": AZURE_LLM_ENDPOINT,
        "OPENAI_API_KEY": AZURE_API_KEY,
        "OPENAI_API_VERSION": AZURE_LLM_API_VERSION,
        "OTEL_SDK_DISABLED": "true",
    }.items():
        os.environ[k] = v

    azure_endpoint = f"{AZURE_LLM_ENDPOINT}/openai/deployments/{deployment}"
    return LLM(
        model=f"azure/{deployment}",
        api_key=AZURE_API_KEY,
        endpoint=azure_endpoint,
        api_version=AZURE_LLM_API_VERSION,
        temperature=0.7,
        timeout=60.0,
        max_retries=3,
    )


# ============================================================================
# LOGGING & APP
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Pre-Consultation Agent Service")

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


# ============================================================================
# REQUEST MODEL
# ============================================================================
class PreConsultationRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="Unique user identifier")
    slot_id: Optional[str] = Field(None)
    input: Optional[str] = Field(None)


# ============================================================================
# CHECKLIST — with dynamic emoji per field
# ============================================================================
CHECKLIST = [
    {
        "num": 1, "label": "Patient's full name", "type": "text", "multi": False,
        "options": [], "emoji": "👤",
        "question": "What is your full name?",
        "valid_hint": "Any name or 'Skip' is acceptable.",
    },
    {
        "num": 2, "label": "Gender", "type": "mcq", "multi": False,
        "options": ["Male", "Female", "Others", "Skip"], "emoji": "🧬",
        "question": "What is your gender?",
        "valid_hint": "Must be one of the provided options.",
    },
    {
        "num": 3, "label": "Main concern or complaint", "type": "text", "multi": False,
        "options": [], "emoji": "🩺",
        "question": "What is your main concern or medical complaint today?",
        "valid_hint": "Any description of a health concern is valid. 'Skip' is also accepted.",
    },
    {
        "num": 4, "label": "Expected result from treatment", "type": "text", "multi": False,
        "options": [], "emoji": "🎯",
        "question": "What outcome are you hoping to achieve from this treatment?",
        "valid_hint": "Any treatment goal is valid. 'Skip' is also accepted.",
    },
    {
        "num": 5, "label": "Medical conditions", "type": "mcq", "multi": True,
        "options": ["Diabetes", "Hypertension", "Asthma", "Hypothyroid", "Seizures", "Blood disorders", "None", "Skip", "Others"],
        "emoji": "🏥",
        "question": "Do you have any pre-existing medical conditions?",
        "valid_hint": "Must be one or more from the provided options.",
    },
    {
        "num": 6, "label": "Past surgical history", "type": "mcq", "multi": False,
        "options": ["Yes", "No", "Skip"], "emoji": "🔪",
        "question": "Have you undergone any surgical procedures in the past?",
        "valid_hint": "Must be Yes, No, or Skip.",
    },
    {
        "num": 7, "label": "History of infectious illnesses", "type": "mcq", "multi": True,
        "options": ["Typhoid", "Chickenpox", "Chikungunya", "Malaria", "Dengue", "Jaundice", "None", "Skip", "Others"],
        "emoji": "🦠",
        "question": "Have you had any infectious illnesses in the past?",
        "valid_hint": "Must be one or more from the provided options.",
    },
    {
        "num": 8, "label": "Multivitamins or supplements usage", "type": "mcq", "multi": False,
        "options": ["Yes", "No", "Skip"], "emoji": "💊",
        "question": "Are you currently taking any multivitamins or supplements?",
        "valid_hint": "Must be Yes, No, or Skip.",
    },
    {
        "num": 9, "label": "Previous hospitalization", "type": "mcq", "multi": False,
        "options": ["Yes", "No", "Skip"], "emoji": "🏨",
        "question": "Have you been hospitalized previously?",
        "valid_hint": "Must be Yes, No, or Skip.",
    },
    {
        "num": 10, "label": "Alcohol habits", "type": "mcq", "multi": True,
        "options": ["Yes", "No", "Occasionally", "Frequently", "Skip", "Others"],
        "emoji": "🍺",
        "question": "How would you describe your alcohol consumption habits?",
        "valid_hint": "Must be one or more from the provided options. Contradictory combinations like 'No' and 'Frequently' are invalid.",
    },
    {
        "num": 11, "label": "Smoking habits", "type": "mcq", "multi": True,
        "options": ["Yes", "No", "Occasionally", "Frequently", "Skip", "Others"],
        "emoji": "🚬",
        "question": "How would you describe your smoking habits?",
        "valid_hint": "Must be one or more from the provided options. Contradictory combinations are invalid.",
    },
    {
        "num": 12, "label": "Food pattern", "type": "mcq", "multi": True,
        "options": ["Veg", "Non-Veg", "Vegan", "Skip"],
        "emoji": "🥗",
        "question": "What best describes your dietary pattern?",
        "valid_hint": "Must be one or more from the provided options.",
    },
    {
        "num": 13, "label": "Sleep pattern", "type": "mcq", "multi": False,
        "options": ["Less than 5 hours", "5-7 hours", "7-9 hours", "More than 9 hours", "Skip"],
        "emoji": "😴",
        "question": "How many hours of sleep do you typically get per night?",
        "valid_hint": "Must be one of the provided options.",
    },
    {
        "num": 14, "label": "Physical activity", "type": "mcq", "multi": False,
        "options": ["No regular exercise", "1-2 days per week", "3-5 days per week", "Daily", "Skip"],
        "emoji": "🏃",
        "question": "How often do you engage in physical activity or exercise?",
        "valid_hint": "Must be one of the provided options.",
    },
    {
        "num": 15, "label": "Body weight in kg", "type": "text", "multi": False,
        "options": [], "emoji": "⚖️",
        "question": "What is your current body weight in kilograms?",
        "valid_hint": "A number (e.g., 70, 70kg) or 'Skip'. Non-numeric, non-weight text is invalid.",
    },
    {
        "num": 16, "label": "Stress level", "type": "mcq", "multi": False,
        "options": ["Low", "Moderate", "High", "Skip"],
        "emoji": "😓",
        "question": "How would you rate your current stress level?",
        "valid_hint": "Must be one of the provided options.",
    },
    {
        "num": 17, "label": "Marital status", "type": "mcq", "multi": False,
        "options": ["Single", "Married", "Divorced", "Widowed", "Skip"],
        "emoji": "💍",
        "question": "What is your marital status?",
        "valid_hint": "Must be one of the provided options.",
    },
    {
        "num": 18, "label": "Menstrual cycle", "type": "mcq", "multi": False,
        "options": ["Regular", "Irregular", "Heavy / Painful", "Menopause", "Others", "Skip"],
        "emoji": "🌸",
        "question": "How would you describe your menstrual cycle?",
        "valid_hint": "Must be one of the provided options.",
    },
    {
        "num": 19, "label": "Family history of medical conditions", "type": "mcq", "multi": True,
        "options": ["Diabetes", "Hypertension", "Heart Disease", "Cancer", "None", "Skip", "Others"],
        "emoji": "👨‍👩‍👧‍👦",
        "question": "Does your family have a history of any medical conditions?",
        "valid_hint": "Must be one or more from the provided options.",
    },
    {
        "num": 20, "label": "Patient verification photo", "type": "upload", "multi": False,
        "options": [], "emoji": "📸",
        "question": "Could you please upload a photo for patient verification?",
        "valid_hint": "A photo upload or 'Skip'.",
    },
]

GREETING_WORDS = {"hello", "hi", "hey", "start", "helo", "hii", "hiii"}


# ============================================================================
# SESSION MEMORY (Python-level, no external DB needed)
# ============================================================================
SESSIONS: Dict[str, Dict] = {}


def get_session(session_id: str) -> Dict:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "history": [],
            "answered": set(),
            "skipped": set(),
            "patient_name": "",
            "gender": "",
            "pending_field_num": 0,  # The field currently awaiting a valid answer
        }
    return SESSIONS[session_id]


def determine_next_field(session: Dict) -> Optional[Dict]:
    answered = session["answered"]
    gender = session.get("gender", "").lower()
    for field in CHECKLIST:
        n = field["num"]
        if n in answered:
            continue
        if n == 18 and "male" in gender and "female" not in gender:
            answered.add(18)
            continue
        return field
    return None


def is_greeting(text: str) -> bool:
    return text.strip().lower() in GREETING_WORDS


def record_answer(session: Dict, user_input: str, field_num: int):
    """Record a validated answer into session memory."""
    normalized = user_input.strip().lower()
    if field_num > 0:
        session["answered"].add(field_num)
        if normalized in {"skip", "prefer not to say", "not willing to disclose"}:
            session["skipped"].add(field_num)
    if field_num == 1 and normalized not in {"skip"}:
        session["patient_name"] = user_input.strip().title()
    if field_num == 2:
        session["gender"] = user_input.strip().lower()


# ============================================================================
# AGENT
# ============================================================================
class PreConsultationAgent:
    def __init__(self):
        self.llm = get_azure_llm_for_crewai()

    def process(self, request: PreConsultationRequest) -> Dict[str, Any]:
        logger.info(f"Session: {request.session_id} | Input: {request.input}")
        session = get_session(request.session_id)
        current_input = (request.input or "").strip()

        # Add user input to history (raw, for context)
        if current_input:
            session["history"].append(f"User: {current_input}")

        # Figure out what field is currently awaiting an answer
        # Use pending_field_num if set (means we re-asked due to invalid answer)
        # Otherwise, determine from answered set
        pending_num = session.get("pending_field_num", 0)
        if pending_num > 0:
            # We're still on the same field — don't auto-advance yet
            current_field = next((f for f in CHECKLIST if f["num"] == pending_num), None)
        else:
            current_field = determine_next_field(session)

        current_field_num = current_field["num"] if current_field else 0
        patient_name = session.get("patient_name", "")
        history_text = "\n".join(session["history"][-40:]) or "Beginning of conversation."

        # If it's a greeting (first message), just ask field 1 - no validation needed
        if is_greeting(current_input) or not current_input:
            next_field = determine_next_field(session)
            session["pending_field_num"] = next_field["num"] if next_field else 0
            task_desc = self._build_question_prompt(session, next_field, history_text, patient_name, None)
            expected = f"JSON asking about: {next_field['label']}" if next_field else "Summary JSON"
        elif current_field is None:
            # All fields answered — generate summary
            task_desc = self._build_summary_prompt(session, history_text, patient_name)
            expected = "JSON with status=end and a detailed clinical booking_summary."
        else:
            # We have a field pending a valid answer — ask LLM to validate it
            task_desc = self._build_validation_and_question_prompt(
                session, current_field, history_text, patient_name, current_input
            )
            expected = f"JSON with answer_accepted bool and either re-asking {current_field['label']} or next question."

        agent = Agent(
            role="HealthBot — Clinical Intake Specialist",
            goal="Conduct a warm, professional patient intake conversation and validate each answer.",
            backstory=(
                "You are HealthBot, a compassionate and detail-oriented medical intake specialist. "
                "You guide patients gently through their intake form, validate their answers, "
                "and produce thorough clinical summaries for the attending physician."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )
        task = Task(description=task_desc, expected_output=expected, agent=agent)
        # NOTE: memory=False (default) — avoids ChromaDB calling standard OpenAI embeddings
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        result = crew.kickoff()

        return self._parse_result(result, session, current_field, current_field_num, current_input)

    # -------------------------------------------------------------------------
    def _build_question_prompt(
        self, session: Dict, next_field: Dict, history_text: str, patient_name: str, prev_answer: Optional[str]
    ) -> str:
        """Build a prompt to simply ask the next question (no validation needed - e.g. first message)."""
        emoji = next_field["emoji"]
        question_text = next_field["question"]
        name_tag = patient_name if patient_name else ""
        ack = f"Hello{', ' + name_tag if name_tag else ''}! Welcome to your pre-consultation. 😊" if not session["history"] or len(session["history"]) <= 1 else ""

        if next_field["type"] == "upload":
            type_note = '"question_type": "upload", "mcq_options": [], "is_multi_select": false'
        elif next_field["type"] == "mcq":
            options_str = json.dumps(next_field["options"])
            is_multi = "true" if next_field["multi"] else "false"
            type_note = f'"question_type": "mcq", "is_multi_select": {is_multi}, "mcq_options": {options_str}'
        else:
            options_str = "[]"
            type_note = '"question_type": "text", "mcq_options": [], "is_multi_select": false'

        mcq_opts = next_field["options"] if next_field["type"] == "mcq" else []

        return f"""You are HealthBot, a warm medical intake assistant 👩‍⚕️.

Conversation so far:
{history_text}

Your task: Ask the patient about their {next_field['label']}.

Formatting rules for your response text:
- Line 1: A warm greeting or acknowledgment (e.g. "{ack or 'Got it! ✅'}")
- Line 2: Empty line
- Line 3: The bold question followed by the emoji: **{question_text}** {emoji}
- Do NOT combine the acknowledgment and question on the same line.
- Do NOT ask any other questions.

Respond with ONLY valid JSON:
{{
  "response": "{ack or 'Got it! ✅'}\\n\\n**{question_text}** {emoji}",
  "question_type": "{'upload' if next_field['type'] == 'upload' else next_field['type']}",
  "mcq_question": "{question_text}",
  "mcq_options": {json.dumps(mcq_opts)},
  "is_multi_select": {'true' if next_field.get('multi') else 'false'},
  "booking_context": "",
  "status": "continue",
  "booking_summary": null,
  "answer_accepted": true,
  "success": true
}}

Note: {type_note}
"""

    # -------------------------------------------------------------------------
    def _build_validation_and_question_prompt(
        self, session: Dict, current_field: Dict, history_text: str, patient_name: str, user_answer: str
    ) -> str:
        """Validate the user's answer to current_field, and if valid, ask the next question. If invalid, re-ask."""
        emoji = current_field["emoji"]
        question_text = current_field["question"]
        valid_hint = current_field["valid_hint"]
        name_tag = patient_name if patient_name else "there"

        if current_field["type"] == "mcq":
            options_str = json.dumps(current_field["options"])
            multi_note = "Multiple selections allowed." if current_field["multi"] else "Single selection only."
            type_note_reask = f'"question_type": "mcq", "is_multi_select": {"true" if current_field["multi"] else "false"}, "mcq_options": {options_str}'
        elif current_field["type"] == "upload":
            options_str = "[]"
            multi_note = ""
            type_note_reask = '"question_type": "upload", "mcq_options": [], "is_multi_select": false'
        else:
            options_str = "[]"
            multi_note = ""
            type_note_reask = '"question_type": "text", "mcq_options": [], "is_multi_select": false'

        # Determine next field label for use if answer IS valid
        answered_copy = session["answered"] | {current_field["num"]}
        gender = session.get("gender", "").lower()
        next_after: Optional[Dict] = None
        for f in CHECKLIST:
            if f["num"] not in answered_copy:
                if f["num"] == 18 and "male" in gender and "female" not in gender:
                    continue
                next_after = f
                break

        if next_after:
            next_emoji = next_after["emoji"]
            next_q = next_after["question"]
            next_options = next_after["options"]
            next_is_multi = next_after.get("multi", False)
            next_type = next_after["type"]
            next_type_note = f'"question_type": "{next_type}", "is_multi_select": {"true" if next_is_multi else "false"}, "mcq_options": {json.dumps(next_options)}'
            next_field_info = f"If the answer IS valid, thank the patient and ask the next question: **{next_q}** {next_emoji}"
        else:
            next_q = ""
            next_emoji = ""
            next_options = []
            next_is_multi = False
            next_type = "text"
            next_type_note = '"question_type": null'
            next_field_info = "If the answer IS valid, all questions are complete. Set status to 'end' and move to summary."

        return f"""You are HealthBot, a warm and professional medical intake assistant 👩‍⚕️.

Conversation so far:
{history_text}

You just asked the patient about: **{current_field['label']}** {emoji}
The patient answered: "{user_answer}"

Your job — VALIDATION FIRST:

Step 1 — Validate the answer:
- Valid answers: {valid_hint}
- Available options (if MCQ): {options_str}. {multi_note}
- Be LENIENT with typos and abbreviations (e.g. "occassionally" = "Occasionally", "diabetis" = "Diabetes"). Treat these as VALID.
- Be STRICT about logically contradictory answers (e.g. selecting both "No" AND "Frequently" for smoking = INVALID).
- Completely unrelated answers (e.g. replying "blue" to a weight question) = INVALID.
- Absurd values for numeric fields (e.g. weight > 500 kg) = INVALID, ask for clarification.
- "Skip" is always valid for any question.

Step 2 — Decide:
- If INVALID: Set "answer_accepted": false. Gently explain the issue in ONE sentence. Then re-ask the SAME question on a NEW line in bold with emoji at end:
  "I noticed [brief issue]. Could you please try again?\\n\\n**{question_text}** {emoji}"
- If VALID: Set "answer_accepted": true. Acknowledge warmly. Then:
  {next_field_info}
  Format: "Got it, {name_tag}! ✅\\n\\n**{next_q}** {next_emoji}" (if there is a next question)

Formatting rules:
- Acknowledgment/explanation on Line 1.
- Empty line (\\n\\n).
- Bold question + emoji at the END of the question on Line 3.
- NEVER put both a validation note and a new question in the same sentence.
- NEVER ask two different questions in one response.

Respond with ONLY valid JSON. Choose ONE of the two templates below:

Template A (answer INVALID - re-ask same question):
{{
  "response": "[gentle note about issue]\\n\\n**{question_text}** {emoji}",
  "question_type": "{current_field['type']}",
  "mcq_question": "{question_text}",
  "mcq_options": {json.dumps(current_field['options'])},
  "is_multi_select": {'true' if current_field.get('multi') else 'false'},
  "booking_context": "",
  "status": "continue",
  "booking_summary": null,
  "answer_accepted": false,
  "success": true
}}

Template B (answer VALID - ask next question):
{{
  "response": "Got it, {name_tag}! ✅\\n\\n**{next_q}** {next_emoji}",
  "question_type": "{next_type}",
  "mcq_question": "{next_q}",
  "mcq_options": {json.dumps(next_options)},
  "is_multi_select": {'true' if next_is_multi else 'false'},
  "booking_context": "",
  "status": "continue",
  "booking_summary": null,
  "answer_accepted": true,
  "success": true
}}
"""

    # -------------------------------------------------------------------------
    def _build_summary_prompt(self, session: Dict, history_text: str, patient_name: str) -> str:
        name_display = patient_name if patient_name else "the patient"
        skipped_fields = session.get("skipped", set())

        # Build skipped fields note for the LLM
        skipped_labels = []
        for field in CHECKLIST:
            if field["num"] in skipped_fields:
                skipped_labels.append(f"- **{field['label']}**: Patient was not willing to disclose.")
        skipped_section = (
            "\n".join(skipped_labels)
            if skipped_labels
            else "No fields were skipped."
        )

        return f"""You are HealthBot, a clinical intake assistant who has just completed collecting information from {name_display}.

Full conversation:
{history_text}

Fields the patient chose to skip (must be mentioned in the summary):
{skipped_section}

Your tasks:
1. Write a warm, personalized closing message to {name_display} thanking them for their time 🙏.
2. Write a detailed clinical intake summary addressed TO THE ATTENDING PHYSICIAN.

Clinical summary instructions:
- Write from the perspective of a medical intake coordinator reporting to a doctor.
- Use professional medical language.
- Divide into 2-3 well-structured paragraphs with clear markdown formatting.
- Bold all important clinical findings, e.g., **Diabetes**, **Hair Transplantation**, **Male**.
- Use relevant emojis sparingly to make the summary visually scannable (e.g., 🩺, ❌, ✅, ⚠️).
- For any field the patient skipped, write: "*Patient was not willing to disclose [field name].*"
- Be thorough and specific — the doctor should not need to re-read the transcript.
- Organize by: Patient Identity → Chief Complaint → Medical History → Lifestyle → Additional Notes.

Respond with ONLY a valid JSON object:
{{
  "response": "Warm, personalized thank-you message to {name_display} with a couple of emojis 🙏✨",
  "question_type": null,
  "mcq_question": "",
  "mcq_options": [],
  "is_multi_select": false,
  "booking_context": "",
  "status": "end",
  "booking_summary": "<detailed markdown clinical summary for the doctor, 2-3 paragraphs, with bold, emojis, and skipped-field notes>",
  "success": true
}}
"""

    # -------------------------------------------------------------------------
    def _parse_result(
        self,
        result,
        session: Dict,
        current_field: Optional[Dict],
        current_field_num: int,
        current_input: str,
    ) -> Dict[str, Any]:
        try:
            result_str = str(result)
            if "```json" in result_str:
                result_str = result_str.split("```json")[1].split("```")[0].strip()
            elif "```" in result_str:
                result_str = result_str.split("```")[1].split("```")[0].strip()

            parsed = json.loads(result_str)
            agent_text = parsed.get("response", result_str)
            session["history"].append(f"Agent: {agent_text}")

            answer_accepted = parsed.get("answer_accepted", True)  # default True for greeting/summary flows

            if answer_accepted and current_field_num > 0 and not is_greeting(current_input):
                # Answer was valid — record it now
                record_answer(session, current_input, current_field_num)
                # Figure out next pending field
                next_f = determine_next_field(session)
                session["pending_field_num"] = next_f["num"] if next_f else 0

                # -------------------------------------------------------
                # KEY FIX: If no more fields remain after recording this
                # answer, we must generate the summary RIGHT NOW in this
                # same request — the LLM validation prompt can't do it.
                # -------------------------------------------------------
                if session["pending_field_num"] == 0:
                    logger.info("All fields answered — generating summary inline.")
                    return self._run_summary_now(session)

            elif not answer_accepted and current_field_num > 0:
                # Answer was invalid — keep pending on same field
                session["pending_field_num"] = current_field_num

            return parsed
        except Exception as e:
            logger.warning(f"JSON parse failed: {e}")
            fallback_text = str(result)
            session["history"].append(f"Agent: {fallback_text}")
            # On parse failure, keep pending field unchanged
            return {
                "response": fallback_text,
                "question_type": current_field["type"] if current_field else "text",
                "mcq_question": current_field["question"] if current_field else "",
                "mcq_options": current_field["options"] if current_field else [],
                "is_multi_select": current_field.get("multi", False) if current_field else False,
                "booking_context": "",
                "status": "continue",
                "booking_summary": None,
                "answer_accepted": True,
                "success": False,
            }

    # -------------------------------------------------------------------------
    def _run_summary_now(self, session: Dict) -> Dict[str, Any]:
        """
        Generate the final clinical summary immediately (called inline after
        the last field is answered). Runs a dedicated LLM call for the summary.
        """
        patient_name = session.get("patient_name", "")
        history_text = "\n".join(session["history"][-60:]) or "Full conversation not available."

        task_desc = self._build_summary_prompt(session, history_text, patient_name)
        expected = "JSON with status=end, a warm response, and a rich markdown booking_summary."

        agent = Agent(
            role="HealthBot — Clinical Intake Specialist",
            goal="Generate a detailed clinical intake summary for the attending physician.",
            backstory=(
                "You are HealthBot, a compassionate and detail-oriented medical intake specialist. "
                "You have just completed the patient intake interview and must now produce a "
                "thorough, doctor-facing clinical summary from the conversation."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )
        task = Task(description=task_desc, expected_output=expected, agent=agent)
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        summary_result = crew.kickoff()

        # Parse the summary response
        try:
            result_str = str(summary_result)
            if "```json" in result_str:
                result_str = result_str.split("```json")[1].split("```")[0].strip()
            elif "```" in result_str:
                result_str = result_str.split("```")[1].split("```")[0].strip()

            summary_parsed = json.loads(result_str)
            session["history"].append(f"Agent: {summary_parsed.get('response', '')}")
            return summary_parsed
        except Exception as e:
            logger.warning(f"Summary JSON parse failed: {e}")
            fallback = str(summary_result)
            session["history"].append(f"Agent: {fallback}")
            return {
                "response": f"Thank you so much, {patient_name or 'dear'}! 🙏 Your intake form is complete. We'll see you at your appointment. ✨",
                "question_type": None,
                "mcq_question": "",
                "mcq_options": [],
                "is_multi_select": False,
                "booking_context": "",
                "status": "end",
                "booking_summary": fallback,
                "answer_accepted": True,
                "success": True,
            }


# ============================================================================
# INIT & ROUTES
# ============================================================================
pre_consult_agent = PreConsultationAgent()


@app.post("/agent/pre-consultation")
async def pre_consultation_endpoint(request: PreConsultationRequest):
    try:
        return pre_consult_agent.process(request)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"status": "ok", "service": "pre-consultation-agent"}

@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(SESSIONS)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
