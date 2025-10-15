import json
import ast
import sqlite3
import re
from typing_extensions import TypedDict
from flask import Flask, request, render_template
from langchain_google_genai import GoogleGenerativeAI

# Removed langchain_ollama import as no longer used
from langgraph.graph import StateGraph, START
from langgraph.errors import GraphRecursionError

# ----------- Schema loading and LLM setup ------------

class State(TypedDict):
    user_query: str
    fields: list
    patient_id: str
    fetched_data: list
    summary: str
    error: str
    sql_query: str 

with open("knowledge_base_clean.json") as f:
    columns_metadata = json.load(f)

FIELD_DEFS = {
    entry["Column Name"]: {
        "description": entry.get("Description", ""),
        "category": entry.get("Category", "N/A")
    }
    for entry in columns_metadata
}

schema_str = "\n".join(
    [
        f"{name}: {info['description']} ({info.get('category', 'N/A')})"
        for name, info in FIELD_DEFS.items()
    ]
)

PATIENT_ID_PROMPT = f"""
You are an expert medical data assistant.

GOAL:
- Extract ONLY the patient ID from the user query.
- The patient ID is a short alphanumeric token identifying the patient.
- If there is no patient ID in the query, respond with None.
- Do NOT output anything else — no explanations, no quotes, no extra text.
- Respond ONLY with the patient ID string or None.

User query:
{{query}}
"""

FIELD_EXTRACTOR_SYSTEM_PROMPT = f"""
You are an expert medical data assistant working with a detailed hospital database schema.

GOAL:
- Extract ALL relevant database field names corresponding to the user's query.
- Consider the full schema and all domain knowledge provided.
- Include fields that match directly or are implied by synonyms or related terms in the query.
- Be cautious NOT to exclude related fields that the user might expect.
- Format the response ONLY as a Python list of exact database field names.
- If no fields match, return an empty list: []

RULES:
- Use only the exact field names from the schema with their definitions.
- Base your mapping strictly on field descriptions, category, and semantic closeness.
- Avoid guessing beyond the schema.

Use the following schema for reference:
{schema_str}

User query:
{{query}}

Return the Python list of matching field names:
"""

# ----------- Gemini LLM wrapper ------------

model = GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="API KEY HERE")

class GeminiLLMWrapper:
    def __init__(self, model):
        self.model = model

    def invoke(self, messages):
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append("System: " + content)
            elif role == "user":
                prompt_parts.append("User: " + content)
            elif role == "assistant":
                prompt_parts.append("Assistant: " + content)
            else:
                prompt_parts.append(content)
        prompt_text = "\n".join(prompt_parts)
        response = self.model.invoke(prompt_text)
        class Response:
            def __init__(self, content):
                self.content = content
        if hasattr(response, "text"):
            return Response(content=response.text)
        else:
            return Response(content=response)

llm = GeminiLLMWrapper(model)

# ----------- Helper functions ------------

def extract_patient_id_llm(user_query: str, llm):
    if not user_query or not user_query.strip():
        return None
    prompt = PATIENT_ID_PROMPT.format(query=user_query)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_query}
    ]
    try:
        result = llm.invoke(messages)
        content = result.content.strip()
        if content.lower() in {"none", "null", "no", ""}:
            return None
        return content
    except Exception:
        return None

def extract_fields(user_query: str, field_defs: dict, llm):
    if not user_query or not user_query.strip():
        return []
    prompt = FIELD_EXTRACTOR_SYSTEM_PROMPT.format(query=user_query)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_query}
    ]
    try:
        result = llm.invoke(messages)
        raw_output = result.content.strip()
        print("Raw field extraction output from LLM:", repr(raw_output))  # debug
        if raw_output.startswith('"') and raw_output.endswith('"'):
            raw_output = raw_output[1:-1].encode("utf-8").decode("unicode_escape")
        extracted = ast.literal_eval(raw_output)
        if isinstance(extracted, list):
            valid_fields = [f for f in extracted if f in field_defs]
            return valid_fields
    except Exception as e:
        print("Field extraction parsing error:", e)
        return []
    return []

table = "patients"
db_file = "database.db"

def validate_sql_query(sql_query: str, patient_id: str, fields: list, table: str) -> tuple[str, bool]:
    sql_upper = sql_query.upper()
    if not sql_upper.startswith("SELECT"):
        return sql_query, False
    if f"Anonymous_Uid = '{patient_id}'" not in sql_query:
        return sql_query, False
    where_clause = sql_query.upper().split("WHERE", 1)
    if len(where_clause) > 1:
        conditions = where_clause[1].split("AND")
        if len(conditions) > 1:
            print("⚠️ Warning: SQL query contains additional filters beyond patient ID")
            print("   This might result in empty results if the filters don't match data")
    return sql_query, True

def summarize_patient_data(data, llm):
    if not data:
        return "No patient data available to summarize."
    
    # Combine all visit data
    combined_text = ""
    for i, record in enumerate(data, start=1):
        combined_text += f"Visit {i}:\n"
        for field, value in record.items():
            if value and str(value).strip():
                display_val = str(value)
                if len(display_val) > 150:
                    display_val = display_val[:147] + "..."
                combined_text += f"  {field}: {display_val}\n"
        combined_text += "\n"
    
    summary_prompt = f"""
You are a medical assistant creating a professional clinical summary.

INSTRUCTIONS:
1. Write the summary as a single, flowing paragraph in narrative format
2. DO NOT use bullet points, numbered lists, or section headers
3. Use complete sentences and proper medical terminology
4. Present information in a logical clinical sequence: findings, diagnoses, then medications/treatment
5. DO NOT include patient identifiers (IDs, names, etc.)
6. DO NOT add information not present in the data
7. DO NOT infer or extrapolate beyond what is explicitly stated
8. Keep the tone formal and professional like a medical record entry

PATIENT DATA:
{combined_text}

Please provide a concise paragraph summary:
"""
    
    messages = [
        {"role": "system", "content": "You are a medical documentation specialist who creates clear, professional clinical summaries."},
        {"role": "user", "content": summary_prompt}
    ]
    
    response = llm.invoke(messages)
    return response.content.strip()



# ----------- Graph nodes with error handling ------------

def node_extract_info(state: State):
    try:
        fields = extract_fields(state["user_query"], FIELD_DEFS, llm)
        patient_id = extract_patient_id_llm(state["user_query"], llm)
        error = ""
        if not fields and not patient_id:
            error = "No fields or patient ID could be extracted."
        return {"fields": fields, "patient_id": patient_id, "error": error}
    except Exception as e:
        return {"fields": [], "patient_id": "", "error": f"Extract info error: {e}"}

def node_validate_and_route(state: State):
    return {}

def route_valid(state: State):
    if state.get("error"):
        return "no_data_node"
    if state.get("fields") and state.get("patient_id"):
        return "fetch_data"
    return "no_data_node"

def node_fetch_data(state: State):
    try:
        user_query = state.get("user_query", "")
        patient_id = state.get("patient_id", "")
        fields = state.get("fields", [])

        if not patient_id or not fields:
            return {"fetched_data": [], "error": "Missing patient ID or fields for fetch", "sql_query": ""}

        prompt = f"""
You are an expert SQL generator for a hospital patient database.

DATABASE SCHEMA:
{schema_str}

TABLE NAME: {table}

TASK:
Generate a complete, executable SQL SELECT query based on the user's request.

REQUIRED INFORMATION:
- Patient ID: {patient_id}
- Relevant fields to retrieve: {fields}

USER QUERY:
{user_query}

CRITICAL INSTRUCTIONS:
1. Generate a syntactically correct SQL query for SQLite
2. Use the exact field names from the provided list: {fields}
3. ALWAYS filter by patient ID: WHERE Anonymous_Uid = '{patient_id}'
4. DO NOT add any additional WHERE conditions or filters unless the user explicitly requests filtering by specific values (e.g., "where diagnosis is X" or "only records with Y > 5")
5. If the user mentions "right eye" or "left eye", they are asking for RIGHT/LEFT EYE FIELDS to be retrieved, NOT asking to filter the data
6. Field names ending in 'Re' are right eye fields, 'Le' are left eye fields - these should be in the SELECT clause, not filtered in WHERE
7. Return ONLY the SQL query without any explanation, markdown formatting, or code blocks
8. The query should be ready to execute directly

SQL Query:
"""

        messages = [
            {"role": "system", "content": "You are a SQL expert. Generate only valid SQL queries without any additional text or formatting."},
            {"role": "user", "content": prompt}
        ]

        sql_response = llm.invoke(messages)
        sql_query = sql_response.content.strip()

        sql_query = re.sub(r"```sql\s*", "", sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r"```\s*", "", sql_query, flags=re.IGNORECASE)
        sql_query = sql_query.strip()

        sql_query = sql_query.rstrip(";") + ";"

        sql_query, is_valid = validate_sql_query(sql_query, patient_id, fields, table)
        if not is_valid:
            return {"fetched_data": [], "error": "Generated SQL query failed validation", "sql_query": sql_query}

        print(f"\n--- Generated SQL Query ---\n{sql_query}\n")

        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute(sql_query)
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description]
        results = [dict(zip(col_names, row)) for row in rows]
        conn.close()

        if not results:
            return {"fetched_data": [], "error": "No data found for this patient.", "sql_query": sql_query}

        return {"fetched_data": results, "error": "", "sql_query": sql_query}
    except sqlite3.Error as e:
        return {"fetched_data": [], "error": f"SQL execution error: {e}", "sql_query": ""}
    except Exception as e:
        return {"fetched_data": [], "error": f"Fetch data error: {e}", "sql_query": ""}

def node_summarize(state: State):
    try:
        data = state.get("fetched_data", [])
        if not data or all(
            all(value is None or str(value).strip() == "" for value in record.values())
            for record in data
        ):
            return {"summary": "No meaningful patient data available to summarize.", "error": ""}
        summary = summarize_patient_data(data, llm)
        return {"summary": summary, "error": ""}
    except Exception as e:
        return {"summary": f"Summarization error: {e}", "error": f"Summarization error: {e}"}

def node_no_data(state: State):
    msg = state.get("error", "")
    if not msg:
        fields_ok = bool(state.get("fields"))
        patient_ok = bool(state.get("patient_id"))
        if not fields_ok and not patient_ok:
            msg = "No relevant fields and Patient ID missing; cannot fetch data."
        elif not fields_ok:
            msg = "No relevant fields extracted from the query."
        elif not patient_ok:
            msg = "Patient ID missing; cannot fetch data."
        else:
            msg = "Unknown error."
    return {"fetched_data": [], "summary": msg, "error": msg}

graph_builder = StateGraph(State)

graph_builder.add_node("extract_info", node_extract_info)
graph_builder.add_node("validate_and_route", node_validate_and_route)
graph_builder.add_node("fetch_data", node_fetch_data)
graph_builder.add_node("summarize", node_summarize)
graph_builder.add_node("no_data_node", node_no_data)

graph_builder.add_edge(START, "extract_info")
graph_builder.add_edge("extract_info", "validate_and_route")

graph_builder.add_conditional_edges(
    "validate_and_route", route_valid,
    {"fetch_data": "fetch_data", "no_data_node": "no_data_node"}
)

graph_builder.add_edge("fetch_data", "summarize")

graph = graph_builder.compile()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        user_query = request.form.get("user_query", "").strip()
        initial_state = {
            "user_query": user_query,
            "fields": [],
            "patient_id": "",
            "fetched_data": [],
            "summary": "",
            "error": "",
            "sql_query": ""
        }
        try:
            final_state = graph.invoke(initial_state, {"recursion_limit": 5})
        except GraphRecursionError:
            final_state = {"error": "Max recursion limit reached; stopping.", "sql_query": ""}
        except Exception as e:
            final_state = {"error": f"Graph execution failed: {e}", "sql_query": ""}
        result = {
            "query":user_query,
            "fields": final_state.get("fields", []),
            "sql_query": final_state.get("sql_query", ""),
            "fetched_data": final_state.get("fetched_data", []),
            "summary": final_state.get("summary", ""),
            "error": final_state.get("error", "")
        }
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
