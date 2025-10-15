# Medical-RAG
A medical RAG Application Using LangGraph


# 🧠 Patient Eye Medication Query System using LangGraph & Gemini API

This project demonstrates how **LLMs and LangGraph** can be integrated with structured healthcare data to enable **intelligent natural language querying** of patient eye medication records.
It takes unstructured user queries, extracts the **patient ID and relevant medical fields**, generates **validated SQL queries**, fetches data from a SQLite database, and finally produces a **summarized clinical report** — all through an automated LangGraph workflow.

---

## ⚙️ Technologies Used

| Component                                   | Purpose                                                                                                      |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **LangGraph**                               | For defining and executing the node-based query flow with conditional routing and recursion handling.        |
| **Gemini 2.5 Flash (Google Generative AI)** | LLM used for prompt-based extraction of patient ID, field names, SQL generation, and clinical summarization. |
| **LangChain Google GenAI Wrapper**          | Integrates Gemini with LangGraph using a structured prompt–response interface.                               |
| **Flask**                                   | Provides a simple web interface for entering user queries and viewing query results/summaries.               |
| **SQLite**                                  | Stores the patient data for fast, file-based access (`database.db`).                                         |
| **JSON**                                    | Holds schema-level knowledge base for field descriptions and categories.                                     |
| **Regex & AST**                             | Used for patient ID validation, field parsing, and SQL post-processing.                                      |

---

## 🧩 Dataset Overview

The dataset contains **eye medication and diagnosis data of patients**, stored in a CSV file.
Each record corresponds to a patient's visit and includes details like:

* **Anonymous_Uid** (Patient ID)
* **Medication details** (Eye drops, dosage, etc.)
* **Ophthalmic findings** (Left/Right eye data fields)
* **Clinical notes, diagnosis, and treatment recommendations**

Additionally, a **data dictionary** was provided — describing each column, its meaning, and its category (diagnosis, medication, eye side, etc.).
This description was converted into a **structured JSON knowledge base** for precise LLM-guided field extraction.

---

## 🧠 Knowledge Base Creation & Usage

### Step 1 — Conversion to JSON

The provided column descriptions were converted to a structured JSON format like:

```json
{
  "Column Name": "Medication_Re",
  "Description": "Right eye medication prescribed",
  "Category": "Medication"
}
```

### Step 2 — Knowledge Base Role

This JSON (`knowledge_base_clean.json`) is used to:

* Dynamically guide the **LLM to map natural language terms** to database column names.
* Improve **field extraction accuracy** through prompt grounding.
* Enable **semantic alignment** between user intent and schema metadata.

---

## 🔄 Data Conversion Workflow

1. **Input**: CSV file containing patient medication and diagnostic data.
2. **Processing**: The CSV is loaded and converted to a **SQLite** database (`database.db`).
3. **Storage**: Each row becomes a record in the `patients` table for fast query execution.

---

## 🧭 LangGraph Workflow Overview

The entire logic is designed as a **LangGraph pipeline** with clear nodes and transitions:

### 🧱 Node Structure

| Node                   | Function                                                          |
| ---------------------- | ----------------------------------------------------------------- |
| **extract_info**       | Extracts patient ID and relevant fields using Gemini LLM.         |
| **validate_and_route** | Checks extraction results and decides whether to proceed or stop. |
| **fetch_data**         | Generates SQL query via LLM → validates → executes on SQLite DB.  |
| **summarize**          | Summarizes fetched patient data into a clinical narrative.        |
| **no_data_node**       | Handles missing/invalid cases gracefully with fallback messages.  |

### 🧩 Conditional Edges & Flow

1. **Start → extract_info**
   Extracts field list and patient ID from the user query.
2. **extract_info → validate_and_route**
   Checks extraction results.
3. **validate_and_route → fetch_data / no_data_node**

   * If valid: generate and execute SQL.
   * Else: show “no data” message.
4. **fetch_data → summarize**
   Converts fetched structured data into a readable medical summary.

All nodes handle errors internally, with recursion limits to prevent infinite loops (`recursion_limit=5`).

---

## 💬 Prompt Engineering Highlights

### 🔹 Patient ID Extraction

A carefully designed prompt ensures:

* Only the **patient ID token** (e.g., `P00123`) is extracted.
* No explanations or additional text.
* Returns `None` if absent.

> This avoids hallucination and ensures ID safety before querying.

### 🔹 Field Extraction

The LLM uses schema-guided context from the JSON knowledge base:

* Matches synonyms and related medical terms.
* Returns a valid Python list of recognized columns.
* Filters against the known schema to prevent invalid fields.

### 🔹 SQL Query Generation

The LLM builds a **SQLite-compatible SQL query**:

* Always filters with `WHERE Anonymous_Uid = '{patient_id}'`.
* Selects only the requested fields.
* Avoids adding extra conditions or markdown formatting.
* Supports “Right/Left eye” logic via `_Re` and `_Le` suffix rules.

### 🔹 Summarization

Fetched records are turned into **narrative clinical summaries** with:

* Formal tone, no identifiers.
* Sequential structure: findings → diagnosis → medications.
* No bullet points or speculative information.

---

## 🧮 Example Flow

**User Input:**

> “Show the right eye medications and diagnosis for patient A102.”

**Pipeline Execution:**

1. Extracts → Patient ID: `A102`, Fields: `[Medication_Re, Diagnosis]`
2. Generates SQL:

   ```sql
   SELECT Medication_Re, Diagnosis FROM patients WHERE Anonymous_Uid = 'A102';
   ```
3. Fetches patient data.
4. Summarizes into a clinical paragraph.

**Final Output:**

> The patient’s right eye was treated with timolol ophthalmic solution for elevated intraocular pressure. The diagnosis was consistent with early glaucoma changes noted during the visit.
---
##  Flow Chart for WorkFlow

<img width="548" height="785" alt="image" src="https://github.com/user-attachments/assets/4a41f87e-ba35-4617-a5a6-b69b12b2a873" />




## 🧰 Error Handling

* **GraphRecursionError**: Stops cyclic executions.
* **SQL validation**: Ensures proper SELECT format and patient ID inclusion.
* **Empty field/patient handling**: Routed to `no_data_node` for clear messages.

---

## 🧾 Summary

This project showcases how **LangGraph + Gemini LLM** can enable:

* Reliable **LLM-driven SQL query automation**
* Context-aware **prompt engineering**
* Graph-based **error-tolerant workflows**
* **Human-like summaries** for structured medical data

It serves as a **blueprint for building intelligent query agents** over healthcare or other domain-specific structured databases.

