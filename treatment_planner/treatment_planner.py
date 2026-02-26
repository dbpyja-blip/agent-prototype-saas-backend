import os
import json
import logging
import re
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Third-party imports
# import psycopg2
# from psycopg2 import pool
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# Load env variables
load_dotenv()
load_dotenv(".env.local", override=True)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# APP & CORS — defined at module level so uvicorn can import it directly
# (allow_origins=["*"] + allow_credentials=True is forbidden by the CORS spec;
#  browsers reject it. Use an explicit origin list instead.)
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "https://healthcareagents.dimensionleap.com",
    "https://dimensionleap-ai-health.vercel.app",
    "http://localhost:3000",
]

app = FastAPI(title="Treatment Planner Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize Router
router = APIRouter()
app.include_router(router)

# --------------------------------------------------------------------------------
# 1. MOCK DATA (Replaces Database)
# --------------------------------------------------------------------------------

MOCK_DB = {
    "clinical_services": [
        {'service_uid': 1, 'service_title': 'Hair Transplant Consultation', 'category_label': 'Dermatology', 'estimated_duration_minutes': 30, 'base_fee': 1500.00},
        {'service_uid': 2, 'service_title': 'PRP Therapy Session', 'category_label': 'Cosmetic Treatment', 'estimated_duration_minutes': 60, 'base_fee': 4500.00},
        {'service_uid': 3, 'service_title': 'Laser Hair Reduction', 'category_label': 'Aesthetic', 'estimated_duration_minutes': 45, 'base_fee': 3000.00},
        {'service_uid': 4, 'service_title': 'General Physician Consultation', 'category_label': 'General Medicine', 'estimated_duration_minutes': 20, 'base_fee': 800.00},
        {'service_uid': 5, 'service_title': 'Skin Allergy Testing', 'category_label': 'Dermatology', 'estimated_duration_minutes': 40, 'base_fee': 2500.00},
        {'service_uid': 6, 'service_title': 'Acne Treatment Program', 'category_label': 'Dermatology', 'estimated_duration_minutes': 50, 'base_fee': 3500.00},
        {'service_uid': 7, 'service_title': 'Dental Cleaning & Polishing', 'category_label': 'Dental', 'estimated_duration_minutes': 45, 'base_fee': 2000.00},
        {'service_uid': 8, 'service_title': 'Root Canal Treatment', 'category_label': 'Dental', 'estimated_duration_minutes': 90, 'base_fee': 6000.00},
        {'service_uid': 9, 'service_title': 'Physiotherapy Session', 'category_label': 'Rehabilitation', 'estimated_duration_minutes': 60, 'base_fee': 1200.00},
        {'service_uid': 10, 'service_title': 'Nutrition Counseling', 'category_label': 'Wellness', 'estimated_duration_minutes': 30, 'base_fee': 1000.00},
        {'service_uid': 11, 'service_title': 'Cardiology Checkup', 'category_label': 'Cardiology', 'estimated_duration_minutes': 35, 'base_fee': 2200.00},
        {'service_uid': 12, 'service_title': 'Orthopedic Evaluation', 'category_label': 'Orthopedics', 'estimated_duration_minutes': 30, 'base_fee': 1800.00},
        {'service_uid': 13, 'service_title': 'Eye Vision Assessment', 'category_label': 'Ophthalmology', 'estimated_duration_minutes': 25, 'base_fee': 900.00},
        {'service_uid': 14, 'service_title': 'Dermatoscopy Examination', 'category_label': 'Dermatology', 'estimated_duration_minutes': 20, 'base_fee': 1300.00},
        {'service_uid': 15, 'service_title': 'Full Body Health Screening', 'category_label': 'Preventive Care', 'estimated_duration_minutes': 120, 'base_fee': 7000.00},
        {'service_uid': 16, 'service_title': 'Post Surgery Follow-up', 'category_label': 'General Medicine', 'estimated_duration_minutes': 20, 'base_fee': 700.00},
        {'service_uid': 17, 'service_title': 'Cosmetic Skin Rejuvenation', 'category_label': 'Aesthetic', 'estimated_duration_minutes': 75, 'base_fee': 5000.00},
        {'service_uid': 18, 'service_title': 'Weight Management Program', 'category_label': 'Wellness', 'estimated_duration_minutes': 60, 'base_fee': 4000.00},
        {'service_uid': 19, 'service_title': 'Thyroid Consultation', 'category_label': 'Endocrinology', 'estimated_duration_minutes': 25, 'base_fee': 1600.00},
        {'service_uid': 20, 'service_title': 'Pediatric Consultation', 'category_label': 'Pediatrics', 'estimated_duration_minutes': 20, 'base_fee': 850.00},
        {'service_uid': 21, 'service_title': 'Vaccination Service', 'category_label': 'Preventive Care', 'estimated_duration_minutes': 15, 'base_fee': 600.00},
        {'service_uid': 22, 'service_title': 'Mental Health Counseling', 'category_label': 'Psychology', 'estimated_duration_minutes': 50, 'base_fee': 2500.00}
    ],
    "medical_products": [
        {'product_uid': 1, 'item_label': 'Minoxidil 5% Solution', 'brand_name': "Dr. Reddy's", 'unit_price': 850.00, 'discounted_rate': 799.00},
        {'product_uid': 2, 'item_label': 'Finasteride Tablets 1mg', 'brand_name': 'Cipla', 'unit_price': 650.00, 'discounted_rate': 599.00},
        {'product_uid': 3, 'item_label': 'Vitamin D3 Capsules', 'brand_name': 'Sun Pharma', 'unit_price': 300.00, 'discounted_rate': 250.00},
        {'product_uid': 4, 'item_label': 'Biotin Supplements', 'brand_name': 'HealthKart', 'unit_price': 500.00, 'discounted_rate': 449.00},
        {'product_uid': 5, 'item_label': 'Hair Growth Serum', 'brand_name': 'Mamaearth', 'unit_price': 999.00, 'discounted_rate': 899.00},
        {'product_uid': 6, 'item_label': 'Anti-Dandruff Shampoo', 'brand_name': 'Head & Shoulders', 'unit_price': 450.00, 'discounted_rate': 399.00},
        {'product_uid': 7, 'item_label': 'Multivitamin Tablets', 'brand_name': 'Himalaya', 'unit_price': 550.00, 'discounted_rate': 499.00},
        {'product_uid': 8, 'item_label': 'Omega 3 Fish Oil', 'brand_name': 'Now Foods', 'unit_price': 1200.00, 'discounted_rate': 1050.00},
        {'product_uid': 9, 'item_label': 'Protein Powder Whey', 'brand_name': 'Optimum Nutrition', 'unit_price': 3500.00, 'discounted_rate': 3200.00},
        {'product_uid': 10, 'item_label': 'Pain Relief Gel', 'brand_name': 'Volini', 'unit_price': 200.00, 'discounted_rate': 180.00},
        {'product_uid': 11, 'item_label': 'Antibiotic Ointment', 'brand_name': 'Neosporin', 'unit_price': 180.00, 'discounted_rate': 160.00},
        {'product_uid': 12, 'item_label': 'SPF 50 Sunscreen Lotion', 'brand_name': 'Neutrogena', 'unit_price': 650.00, 'discounted_rate': 599.00},
        {'product_uid': 13, 'item_label': 'Salicylic Acid Face Wash', 'brand_name': 'Minimalist', 'unit_price': 349.00, 'discounted_rate': 299.00},
        {'product_uid': 14, 'item_label': 'Collagen Powder', 'brand_name': 'Oziva', 'unit_price': 2200.00, 'discounted_rate': 1999.00},
        {'product_uid': 15, 'item_label': 'Zinc Tablets', 'brand_name': 'HealthVit', 'unit_price': 250.00, 'discounted_rate': 220.00},
        {'product_uid': 16, 'item_label': 'Iron Supplements', 'brand_name': 'Dexorange', 'unit_price': 190.00, 'discounted_rate': 170.00},
        {'product_uid': 17, 'item_label': 'Aloe Vera Gel', 'brand_name': 'Patanjali', 'unit_price': 120.00, 'discounted_rate': 99.00},
        {'product_uid': 18, 'item_label': 'Ketoconazole Shampoo', 'brand_name': 'Nizoral', 'unit_price': 700.00, 'discounted_rate': 650.00},
        {'product_uid': 19, 'item_label': 'Glucose Monitoring Strips', 'brand_name': 'Accu-Chek', 'unit_price': 1500.00, 'discounted_rate': 1400.00},
        {'product_uid': 20, 'item_label': 'Therapeutic Hair Mask', 'brand_name': "L'Oreal", 'unit_price': 850.00, 'discounted_rate': 799.00},
        {'product_uid': 21, 'item_label': 'Probiotic Capsules', 'brand_name': 'GNC', 'unit_price': 1600.00, 'discounted_rate': 1450.00},
        {'product_uid': 22, 'item_label': 'Calcium Tablets', 'brand_name': 'Shelcal', 'unit_price': 300.00, 'discounted_rate': 270.00}
    ],
    "lab_diagnostic_catalog": [
        {'test_uid': 1, 'examination_name': 'Complete Blood Count (CBC)', 'department_tag': 'Pathology', 'test_cost': 350.00},
        {'test_uid': 2, 'examination_name': 'Lipid Profile Test', 'department_tag': 'Biochemistry', 'test_cost': 800.00},
        {'test_uid': 3, 'examination_name': 'Liver Function Test (LFT)', 'department_tag': 'Biochemistry', 'test_cost': 900.00},
        {'test_uid': 4, 'examination_name': 'Kidney Function Test (KFT)', 'department_tag': 'Biochemistry', 'test_cost': 850.00},
        {'test_uid': 5, 'examination_name': 'Thyroid Profile (T3 T4 TSH)', 'department_tag': 'Endocrinology', 'test_cost': 750.00},
        {'test_uid': 6, 'examination_name': 'Blood Sugar Fasting', 'department_tag': 'Diabetology', 'test_cost': 120.00},
        {'test_uid': 7, 'examination_name': 'HbA1c Test', 'department_tag': 'Diabetology', 'test_cost': 600.00},
        {'test_uid': 8, 'examination_name': 'Vitamin B12 Test', 'department_tag': 'Pathology', 'test_cost': 950.00},
        {'test_uid': 9, 'examination_name': 'Vitamin D Test', 'department_tag': 'Pathology', 'test_cost': 1200.00},
        {'test_uid': 10, 'examination_name': 'Urine Routine Examination', 'department_tag': 'Pathology', 'test_cost': 200.00},
        {'test_uid': 11, 'examination_name': 'ESR Test', 'department_tag': 'Pathology', 'test_cost': 150.00},
        {'test_uid': 12, 'examination_name': 'CRP Test', 'department_tag': 'Immunology', 'test_cost': 700.00},
        {'test_uid': 13, 'examination_name': 'Dengue NS1 Antigen Test', 'department_tag': 'Microbiology', 'test_cost': 1000.00},
        {'test_uid': 14, 'examination_name': 'Malaria Parasite Test', 'department_tag': 'Microbiology', 'test_cost': 500.00},
        {'test_uid': 15, 'examination_name': 'COVID-19 RT-PCR', 'department_tag': 'Virology', 'test_cost': 1500.00},
        {'test_uid': 16, 'examination_name': 'X-Ray Chest', 'department_tag': 'Radiology', 'test_cost': 800.00},
        {'test_uid': 17, 'examination_name': 'Ultrasound Abdomen', 'department_tag': 'Radiology', 'test_cost': 1800.00},
        {'test_uid': 18, 'examination_name': 'ECG Test', 'department_tag': 'Cardiology', 'test_cost': 400.00},
        {'test_uid': 19, 'examination_name': '2D Echo Test', 'department_tag': 'Cardiology', 'test_cost': 2500.00},
        {'test_uid': 20, 'examination_name': 'Allergy Panel Test', 'department_tag': 'Immunology', 'test_cost': 2200.00},
        {'test_uid': 21, 'examination_name': 'Hormone Panel Test', 'department_tag': 'Endocrinology', 'test_cost': 3000.00},
        {'test_uid': 22, 'examination_name': 'Ferritin Test', 'department_tag': 'Pathology', 'test_cost': 900.00}
    ]
}

# --------------------------------------------------------------------------------
# 2. MEMORY IMPLEMENTATION (Simple Fallback)
# --------------------------------------------------------------------------------

class SimpleMemory:
    """Simple in-memory storage for session history"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SimpleMemory, cls).__new__(cls)
            cls._instance.store = {} # {session_id: [messages]}
        return cls._instance

    def add(self, message, user_id=None, metadata=None):
        session_id = metadata.get('session_id', 'default') if metadata else 'default'
            
        if session_id not in self.store:
            self.store[session_id] = []
        
        entry = {
            "memory": message,
            "user_id": user_id,
            "metadata": metadata
        }
        self.store[session_id].append(entry)
        # Limit memory to last 20 turns
        if len(self.store[session_id]) > 20:
             self.store[session_id] = self.store[session_id][-20:]

    def get_all(self, user_id=None):
        all_mem = []
        for sess_id, msgs in self.store.items():
            all_mem.extend(msgs)
        return all_mem
        
    def get_session_history(self, session_id: str) -> List[Dict]:
        return self.store.get(session_id, [])

# Use a global instance
memory_store = SimpleMemory()

# --------------------------------------------------------------------------------
# 3. AZURE OPENAI CLIENT (Inlined from azure_openai_utils.py)
# --------------------------------------------------------------------------------

def get_azure_llm_for_crewai(model_name: Optional[str] = None):
    """
    Create and return a CrewAI LLM object configured for Azure OpenAI.
    Matches logic from azure_openai_utils.py
    """
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_LLM_ENDPOINT = os.getenv("AZURE_LLM_ENDPOINT") or os.getenv("AZURE_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
    if AZURE_LLM_ENDPOINT:
        AZURE_LLM_ENDPOINT = AZURE_LLM_ENDPOINT.rstrip("/")
    AZURE_LLM_API_VERSION = "2024-02-01"
    AZURE_LLM_DEPLOYMENT = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4o-mini")
    
    deployment = model_name or AZURE_LLM_DEPLOYMENT
    
    if not AZURE_API_KEY:
        raise ValueError("AZURE_API_KEY environment variable is not set")
    if not AZURE_LLM_ENDPOINT:
        raise ValueError("AZURE_LLM_ENDPOINT environment variable is not set")
    
    # Set environment variables that libraries looking for
    os.environ["AZURE_API_KEY"] = AZURE_API_KEY
    os.environ["AZURE_API_BASE"] = AZURE_LLM_ENDPOINT
    os.environ["AZURE_ENDPOINT"] = AZURE_LLM_ENDPOINT
    os.environ["AZURE_API_VERSION"] = AZURE_LLM_API_VERSION
    os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_LLM_ENDPOINT
    os.environ["AZURE_OPENAI_API_KEY"] = AZURE_API_KEY
    
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_BASE"] = AZURE_LLM_ENDPOINT
    os.environ["OPENAI_API_KEY"] = AZURE_API_KEY
    os.environ["OPENAI_API_VERSION"] = AZURE_LLM_API_VERSION
    
    os.environ["OTEL_SDK_DISABLED"] = "true"
    
    azure_endpoint = f"{AZURE_LLM_ENDPOINT}/openai/deployments/{deployment}"
    
    return LLM(
        model=f"azure/{deployment}",
        api_key=AZURE_API_KEY,
        endpoint=azure_endpoint,
        api_version=AZURE_LLM_API_VERSION,
        temperature=0.7,
        timeout=60.0,
        max_retries=3
    )

# --------------------------------------------------------------------------------
# 4. DATABASE QUERY TOOL (Inlined & Simplified)
# --------------------------------------------------------------------------------

class DatabaseQueryToolSchema(BaseModel):
    """Schema for DatabaseQueryTool parameters"""
    query: str = Field(description="SQL SELECT query or search term for database")
    query_type: str = Field(default="general", description="Type of query: 'doctors', 'services', 'bookings', or 'general'")
    
    @field_validator('query', mode='before')
    def extract_query_string(cls, v):
        if isinstance(v, list) and len(v) > 0:
            if isinstance(v[0], dict) and 'query' in v[0]:
                return v[0]['query']
            return str(v[0])
        if isinstance(v, dict):
            if 'query' in v:
                return v['query']
            return str(v)
        return str(v)

class DatabaseQueryTool(BaseTool):
    name: str = "database_query"
    description: str = "Query the database. MANDATORY: For specific searches, provide a complete SQL SELECT query."
    args_schema: type = DatabaseQueryToolSchema
    
    def _run(self, query: str, query_type: str = "general") -> str:
        """Execute mock database query"""
        try:
            query = str(query).strip() if query else ""
            if not query:
                return "Empty query provided"
            
            # Simple keyword search simulation based on standard SQL patterns
            query_lower = query.lower()
            
            # 1. Identify Target Table
            target_table = None
            
            # Try from SQL
            if "from clinical_services" in query_lower:
                target_table = "clinical_services"
            elif "from medical_products" in query_lower:
                target_table = "medical_products"
            elif "from lab_diagnostic_catalog" in query_lower:
                target_table = "lab_diagnostic_catalog"
            
            # Try from query_type if not found in SQL
            if not target_table:
                if "service" in query_type.lower():
                    target_table = "clinical_services"
                elif "product" in query_type.lower() or "medication" in query_type.lower():
                    target_table = "medical_products"
                elif "lab" in query_type.lower() or "diagnostic" in query_type.lower():
                    target_table = "lab_diagnostic_catalog"
            
            if not target_table:
                return "Table not found or supported in mock DB. Please specify query_type (services, products, labs)."

            # 2. Extract Search Term
            search_term = query
            
            # If it looks like SQL, extract the term from ILIKE
            if "ilike" in query_lower:
                parts = query_lower.split("ilike")
                if len(parts) > 1:
                    raw_term = parts[1].split("limit")[0].strip()
                    search_term = raw_term.replace("'", "").replace("%", "").replace(";", "")
            else:
                # Require explicit SELECT if not using query_type fallback, but here we allow loose queries
                # Clean up if the agent just passed "SELECT * FROM ... WHERE ... 'term'" awkwardly
                if "select" in query_lower and "where" in query_lower:
                     # Fallback extraction roughly
                     pass
            
            search_term = search_term.replace("%", "").replace("'", "").strip()
            
            if not search_term:
                return "Could not parse search term"

            # 3. Filter Data with Scoring
            scored_results = []
            source_data = MOCK_DB.get(target_table, [])
            
            # Tokenize search term
            search_tokens = [t for t in search_term.lower().split() if len(t) > 2] # Ignore short words
            
            for item in source_data:
                score = 0
                item_str = str(item).lower()
                
                # Full substring match gets highest priority
                if search_term.lower() in item_str:
                    score += 10
                
                # Token match
                for token in search_tokens:
                    if token in item_str:
                        score += 1
                
                if score > 0:
                    scored_results.append((score, item))
            
            # Sort by score desc and take top 5
            scored_results.sort(key=lambda x: x[0], reverse=True)
            results = [item for score, item in scored_results[:5]]
            
            # 4. Format results
            if not results:
                return "No results found for the query"
                
            result_str = "Query Results:\n"
            for i, row in enumerate(results[:5]): # Limit 5
                result_str += f"Row {i+1}: {row}\n"
            
            return result_str
                    
        except Exception as e:
            logger.error(f"DatabaseQueryTool Error: {e}", exc_info=True)
            return f"Error executing database query: {str(e)}"
    
    def _execute_simple_query(self, sql_query: str) -> str:
        # Redirect all to _run since we aren't doing real SQL
        return self._run(sql_query)

# --------------------------------------------------------------------------------
# 5. TREATMENT PLANNER ENDPOINT LOGIC
# --------------------------------------------------------------------------------

class TreatmentPlannerRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier for this planning session.")
    user_id: str = Field(..., description="Unique user/doctor identifier.")
    treatment_text: Optional[str] = Field(None, description="The doctor's INITIAL notes describing the treatment plan.")
    input: Optional[str] = Field(None, description="Doctor's feedback or modification instructions.")
    slot_id: str = Field(default="reference_only", description="Slot ID for reference (not used for querying).")

class TreatmentPlannerResponse(BaseModel):
    treatment_plans: List[Dict[str, Any]]
    patient_info: Dict[str, Any] = {}
    status: str = Field(..., description="Status of the planning process: 'in_progress' or 'finished'.")
    message: str = Field(..., description="Agent's response message to the doctor.")
    success: bool
    error: Optional[str] = None

@router.post("/agent/treatment-planner", response_model=TreatmentPlannerResponse)
async def run_treatment_planner(request: TreatmentPlannerRequest):
    """
    Agentic Treatment Planner (Interactive):
    - Parses doctor's notes or feedbacks.
    - Maintains session history using SimpleMemory.
    - Verifies services, products, and lab tests against the database.
    - Structures the output into Treatment Plans (A, B, C).
    - Status loops: 'in_progress' -> 'finished'.
    """
    logger.info(f"Processing Treatment Planner Request (Session: {request.session_id}, Slot ID: {request.slot_id})")
    
    try:
        # 1. Retrieve History
        history = memory_store.get_session_history(request.session_id)
        
        # 2. Determine Mode (Initial vs Edit)
        current_input = request.treatment_text if request.treatment_text else request.input
        if not current_input:
             # If no new input, validation error or just return state (handled below)
             # But request model allows optional.
             if not history:
                 return TreatmentPlannerResponse(
                     treatment_plans=[], 
                     status="in_progress", 
                     message="Please provide initial treatment text.", 
                     success=False
                 )
             current_input = "(No new input, please review current state)"

        # Store User Input in Memory
        memory_store.add(f"DOCTOR: {current_input}", user_id=request.user_id, metadata={"session_id": request.session_id})
        
        # Format History for Prompt
        formatted_history = "\n".join([m['memory'] for m in history[-10:]])
        
        # 3. Initialize Agent
        db_tool = DatabaseQueryTool()
        
        planner_agent = Agent(
            role='Treatment Planner',
            goal='Create and Refine Verified Treatment Plans (A, B, C) based on Doctor\'s Input',
            backstory=f"""You are an intelligent Treatment Planner Assistant. You collaborate with a doctor to build perfect Treatment Plans.

**YOUR CORE CAPABILITY**:
1. You maintain a structured Plan (Plan A, B, C).
2. You accept "Edit Commands" from the doctor (e.g., "Change A to 3000 grafts", "Remove Plan B").
3. You ALWAYS verify every medical item against the database.

**DATABASE RULES (STRICT)**:
- Always use SQL SELECT queries with ILIKE for fuzzy search.

1. Services Table: `clinical_services`
   - Columns:
     - service_uid (primary key)
     - service_title (service name)
     - category_label (department/category)
     - estimated_duration_minutes (duration)
     - base_fee (service price)
   - Query Example:
     SELECT service_uid, service_title, category_label, base_fee 
     FROM clinical_services 
     WHERE service_title ILIKE '%keyword%' 
     LIMIT 5;

2. Products Table: `medical_products`
   - Columns:
     - product_uid (primary key)
     - item_label (product/medication name)
     - brand_name (brand)
     - unit_price (original price)
     - discounted_rate (discount price)
   - Query Example:
     SELECT product_uid, item_label, brand_name, unit_price, discounted_rate 
     FROM medical_products 
     WHERE item_label ILIKE '%keyword%' 
     LIMIT 5;

3. Lab Tests Table: `lab_diagnostic_catalog`
   - Columns:
     - test_uid (primary key)
     - examination_name (test name - IMPORTANT: use this for lookup)
     - department_tag (department)
     - test_cost (price)
   - Query Example:
     SELECT test_uid, examination_name, test_cost 
     FROM lab_diagnostic_catalog 
     WHERE examination_name ILIKE '%keyword%' 
     LIMIT 5;

**VERIFICATION & OUTPUT RULES**:
1. **Verified Status**: 
   - If found in DB -> "verified": true.
   - If NOT found -> "verified": false (do NOT use null).
2. **Medications**: If a medication is not found, `product_uid` is null and `verified` is false.
3. **Lab Tests**: Ensure you use `examination_name` column for lookups. If not found, `test_uid` is null and `verified` is false.
4. **Services**: Ensure you use `service_title` column for lookups. If not found, `service_uid` is null and `verified` is false.

**INTERACTION FLOW**:
- If this is the START: Parse the notes -> Create Plans -> Status: 'in_progress'. ask "Does this look correct?"
- If the doctor gives FEEDBACK: **START FROM THE LAST FULL PLAN IN HISTORY. Only change what the doctor explicitly asked. Copy everything else EXACTLY as it was — do NOT drop or reset any medications, lab tests, or other fields.** Status: 'in_progress'. ask "Any other changes?"
- If the doctor says "Looks good" or "Confirm" or "Finish": Status: 'finished'.

**SERVICE MATCHING RULE (CRITICAL)**:
- When you search for a service and the DB returns results, pick the BEST MATCHING result from the list.
- Do NOT return null/unverified just because the exact name typed by the doctor was not found. Use the closest DB match.
- Example: Doctor says "Hair Transplantation" -> DB returns "Hair Transplant Consultation" -> Use that entry, mark verified: true.
- Only set verified: false if NO related result is found at all.
**HALLUCINATION PREVENTION PROTOCOL (CRITICAL)**:
1. **NO GUESSING**: If a medication, service, or lab test is not found in the database, you MUST set `"verified": false` and leave fields like `product_uid`, `test_uid` as `null`.
2. **NO INVENTED DATA**: Do NOT make up prices, brand names, or codes. Only use what the `database_query` tool returns.
3. **NO ASSUMPTIONS**: If the doctor does not specify a dosage, frequency, or duration, do NOT assume "1 tablet" or "daily". use the strict placeholders "Enter dosage", etc.
4. **STRICT VERIFICATION**: You are a *filter*, not a creative writer. If the user asks for "Magic Pill", and the DB says "No results", you must return it as unverified, not invent a "Magic Pill 500mg".

**OUTPUT FORMAT (STRICT)**:
You must return a JSON object with:
- `treatment_plans`: The current valid list of plans.
- `status`: "in_progress" or "finished".
- `message`: A short polite text to the doctor.

**TREATMENT PLAN STRUCTURE (STRICT)**:
Each plan in `treatment_plans` MUST follow this exact structure:
```json
{{
  "plan": "A",
  "details": {{
    "service": {{ ... }},
    "grafts": 2500, // Include if applicable
    "medications": [
      {{
        "product_uid": 123,
        "item_label": "Minoxidil 5%",
        "brand_name": "Mintop",
        "unit_price": 500,
        "discounted_rate": 450,
        "verified": true,
        "dosage": "1ml", // MUST be present. If unknown, use "Enter dosage"
        "frequency": "Twice daily", // MUST be present. If unknown, use "Enter frequency"
        "duration": "6 months", // MUST be present. If unknown, use "Enter duration"
        "route": "Topical" // MUST be present. If unknown, use "Enter route"
      }}
    ],
    "lab_tests": [ ... ]
  }}
}}
```

**CRITICAL RULES FOR EMPTY FIELDS**:
1. **NEVER** leave `dosage`, `frequency`, `duration`, or `route` as empty strings ("") or null.
2. If the doctor did not specify these details, you **MUST** use the following placeholders:
   - "Enter dosage"
   - "Enter frequency"
   - "Enter duration"
   - "Enter route"
3. This ensures the UI displays a prompt for the user to fill them in, rather than a blank space.
""",
            verbose=True,
            allow_delegation=False,
            tools=[db_tool],
            llm=get_azure_llm_for_crewai(),
            max_iter=15
        )

        task_description = f"""
CURRENT SESSION HISTORY:
```
{formatted_history}
```

LATEST INPUT FROM DOCTOR:
"{current_input}"

YOUR TASK:
1. Analyze the history and latest input.
2. If this is a NEW plan (no history), extract and verify all items from the doctor's notes.
3. If this is an EDIT:
   a. FIRST, extract the LAST FULL PLAN JSON from the session history above.
   b. USE that plan as your BASE. Copy it exactly.
   c. ONLY modify the specific fields the doctor explicitly asked to change.
   d. CARRY OVER all other fields unchanged — medications, lab_tests, grafts, services — everything the doctor did NOT ask to change must remain IDENTICAL to the previous plan.
   e. VERIFY only the newly added/changed items against the database.
   f. Do NOT drop, remove, or reset any field unless the doctor explicitly asked to remove it.
4. SERVICE MATCHING: When searching for a service, pick the BEST MATCH from the DB results. Do not return null if a close match exists.
5. Determine if the doctor is satisfied ("finished") or still editing ("in_progress").

RETURN JSON ONLY:
{{
  "treatment_plans": [ ... ],
  "status": "in_progress" | "finished",
  "message": "...",
  "patient_info": {{}},
  "success": true
}}
"""
        
        planning_task = Task(
            description=task_description,
            expected_output="Pure JSON object with treatment_plans, status, and message. NO markdown.",
            agent=planner_agent
        )

        # 4. Execute Crew
        crew = Crew(
            agents=[planner_agent],
            tasks=[planning_task],
            verbose=True,
            process=Process.sequential
        )

        result = crew.kickoff()
        
        # 5. Parse & Store Result
        result_str = str(result)
        if "```json" in result_str:
            result_str = result_str.split("```json")[1].split("```")[0].strip()
        elif "```" in result_str:
            result_str = result_str.split("```")[1].split("```")[0].strip()
        
        result_str = result_str.replace("True", "true").replace("False", "false").replace("None", "null")
        parsed_result = json.loads(result_str)
        
        # Add Agent Response to Memory — store FULL plan JSON so edits have complete context
        full_plans_json = json.dumps(parsed_result.get('treatment_plans', []))
        agent_msg = f"AGENT (Status: {parsed_result.get('status')}): {parsed_result.get('message')}\nPlans: {full_plans_json}"
        memory_store.add(agent_msg, user_id=request.user_id, metadata={"session_id": request.session_id})
        
        return parsed_result

    except Exception as e:
        logger.error(f"Error in treatment planner: {e}")
        return {
            "treatment_plans": [],
            "status": "error",
            "message": f"System error: {str(e)}",
            "success": False,
            "error": str(e)
        }

# Health-check endpoint so Render's GET / returns 200 instead of 404
# (a persistent 404 on the health check can cause Render to mark the service unhealthy)
@app.get("/")
async def health_check():
    return {"status": "ok", "service": "treatment-planner"}

if __name__ == "__main__":
    import uvicorn
    # Read PORT from env (Render injects it); fall back to 8002 for local dev
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
