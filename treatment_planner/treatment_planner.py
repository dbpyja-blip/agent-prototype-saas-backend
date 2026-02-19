import os
import json
import logging
import re
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Third-party imports
import psycopg2
from psycopg2 import pool
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# Load env variables
load_dotenv()
load_dotenv(".env.local", override=True)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Router
router = APIRouter()

# --------------------------------------------------------------------------------
# 1. DATABASE CONNECTION POOL (Inlined from statictools.py)
# --------------------------------------------------------------------------------

_sync_connection_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None

def init_sync_connection_pool(minconn: int = 2, maxconn: int = 10):
    """Initialize synchronous connection pool for psycopg2"""
    global _sync_connection_pool
    if _sync_connection_pool is None:
        try:
            _sync_connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=minconn,
                maxconn=maxconn,
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT", 5432)
            )
            logger.info(f"Sync connection pool initialized: min={minconn}, max={maxconn}")
        except Exception as e:
            logger.error(f"Failed to initialize sync connection pool: {e}")
            raise

class PooledConnection:
    """Wrapper for pooled connections that returns to pool on close"""
    def __init__(self, conn, pool):
        self._conn = conn
        self._pool = pool
        self._closed = False
    
    def __enter__(self):
        return self._conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Return connection to pool instead of closing it"""
        if not self._closed and self._pool:
            try:
                self._pool.putconn(self._conn)
                self._closed = True
            except Exception as e:
                logger.warning(f"Error returning connection to pool: {e}")
    
    def cursor(self, *args, **kwargs):
        return self._conn.cursor(*args, **kwargs)
    
    def commit(self):
        return self._conn.commit()
    
    def rollback(self):
        return self._conn.rollback()
    
    def __getattr__(self, name):
        return getattr(self._conn, name)

def _connect():
    """Get connection from sync pool - automatically returns to pool when closed"""
    global _sync_connection_pool
    try:
        if _sync_connection_pool is None:
            init_sync_connection_pool()
        
        conn = _sync_connection_pool.getconn()
        return PooledConnection(conn, _sync_connection_pool)
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        # Fallback to direct connection if pool fails
        return psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", 5432)
        )

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
        """Execute database query"""
        try:
            query = str(query).strip() if query else ""
            if not query:
                return "Empty query provided"
            
            # Direct SQL execution
            if query.lstrip().lower().startswith("select"):
                return self._execute_simple_query(query)
            
            # Fallback for non-SQL queries
            return "Please provide a valid SQL SELECT query."
                    
        except Exception as e:
            logger.error(f"DatabaseQueryTool Error: {e}", exc_info=True)
            return f"Error executing database query: {str(e)}"

    def _execute_simple_query(self, sql_query: str) -> str:
        """Execute simple SQL query using direct database connection"""
        try:
            if not sql_query.lstrip().lower().startswith("select"):
                return "Only SELECT queries are allowed"

            with _connect() as conn, conn.cursor() as cur:
                cur.execute(sql_query)
                data = cur.fetchall()
                
                try:
                    column_names = [desc[0] for desc in cur.description] if cur.description else []
                except:
                    column_names = []
                
                if not data:
                    return "No results found for the query"
                
                if len(data) == 1 and len(data[0]) == 1:
                    return f"Query Result: {data[0][0]}"
                else:
                    result_str = "Query Results:\n"
                    # Limit rows to avoid token overflow
                    for i, row in enumerate(data[:15]): 
                        if column_names and len(row) == len(column_names):
                            row_dict = dict(zip(column_names, row))
                            result_str += f"Row {i+1}: {row_dict}\n"
                        else:
                            result_str += f"Row {i+1}: {row}\n"
                    if len(data) > 15:
                        result_str += f"... and {len(data) - 15} more rows"
                    return result_str
                    
        except Exception as e:
            return f"Error executing query: {str(e)}"

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
- If the doctor gives FEEDBACK: Modify the plans -> Status: 'in_progress'. ask "Any other changes?"
- If the doctor says "Looks good" or "Confirm" or "Finish": Status: 'finished'.
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
2. If this is a new plan, extract and verify everything.
3. If this is an edit, modify the *existing* plans from history accordingly.
4. VERIFY all added/modified items against the database.
   - **Labs**: Use `test_name` for lookup.
   - **Status**: If any item is not found, set `"verified": false` (NOT null).
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
        
        # Add Agent Response to Memory
        agent_msg = f"AGENT (Status: {parsed_result.get('status')}): {parsed_result.get('message')}\nPlans: {json.dumps(parsed_result.get('treatment_plans', []))[:200]}..." 
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

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    # Make sure pool is initialized if running directly
    init_sync_connection_pool()
    
    app = FastAPI()
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins - restrict in production
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods including OPTIONS, POST, etc.
        allow_headers=["*"],  # Allows all headers
    )
    
    app.include_router(router)
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
