# AI Research Paper & Architecture Literature Assistant (Project 7)

A modular, production-grade Retrieval-Augmented Generation (RAG) system engineered to parse, index, and strictly answer technical queries from deep-learning research literature (`transformer_rag_research_paper_50p.pdf`). 

Built with **Python**, **ChromaDB**, **PyPDF**, and **Google Gemini**, with local persistence and zero-hallucination out-of-domain rejection.

---

## Architecture & Workflow

```
                                  +------------------------------------+
                                  | transformer_rag_research_paper.pdf |
                                  +------------------------------------+
                                                    |
                                                    v
                                      [ pypdf Page-by-Page Parser ]
                                        (1-indexed page tracking)
                                                    |
                                                    v
                                  [ Sliding-Window Character Chunking ]
                                   (800 chars / 150 character overlap)
                                                    |
                                                    v
                                     [ ChromaDB Persistent Storage ]
                                     (path: ./chroma_db on local disk)
                                                    |
                                                    v
                                     +-------------------------------+
                                     |  Collection count > 0 check   |
                                     |  Reuses index on next runs    |
                                     +-------------------------------+
                                                    |
   User Question ------------------------> [ Dense Vector Query ] (Top-4)
                                                    |
                                                    v
                                       [ Context Assembly & Prompt ]
                                                    |
                                                    v
                                      [ Gemini / Strict Grounding ]
                                      (temperature=0.0 / Zero Guess)
                                                    |
                       +----------------------------+----------------------------+
                       |                                                         |
             [ Answer Found in Chunks ]                                [ Not Found in Chunks ]
                       |                                                         |
                       v                                                         v
         {"answer": "...", "sources": [...]}                      {"answer": "[not found]", "sources": []}
```

### 1. Ingestion & 1-Indexed Page Extraction
- Utilizes `pypdf.PdfReader` to extract textual content sequentially across all 50 pages.
- Page numbering is strictly **1-indexed** (`page_num = page_idx + 1`) to match human-readable publication citations and document footers.

### 2. Overlapping Chunking Strategy
- Each page is decomposed into sliding-window text segments of **~800 characters** with **150 characters of overlap** (`step = 650`).
- **Why overlap matters (VIVA topic):** Technical definitions, hyperparameter values, and table rows often cross chunk boundaries. Overlap ensures phrases are never split mid-sentence, preserving complete semantic continuity.
- Rich metadata is tagged to every chunk:
  ```python
  {
      "page": page_num,             # e.g., 16
      "chunk_id": f"page_{page_num}_chunk_{idx}"  # e.g., "page_16_chunk_0"
  }
  ```

### 3. Persistent Local Vector Store (`ChromaDB`)
- Employs `chromadb.PersistentClient(path="./chroma_db")` to serialize embeddings and metadata to disk.
- **Fast Startup / Index Reuse:** Before chunking or embedding, the system checks `collection.count() > 0`. If embeddings already exist on disk, ingestion is bypassed, providing sub-second query readiness.
- Runs with local ONNX vector embeddings (`all-MiniLM-L6-v2`) without relying on network calls for vector queries.

### 4. Strict Grounding Prompt Engineering
- Queries are passed to Google Gemini with `temperature=0.0` for deterministic, hallucination-free output.
- The prompt instructs the model that it must **only** state facts explicitly provided in the retrieved context blocks. If the context does not contain the answer, the model is strictly constrained to output **`[not found]`**.

---

## Core Interface

The pipeline exposes the exact standardized interface required by Project 7:

```python
def answer(question: str) -> dict:
    """
    Takes a natural-language question, queries ChromaDB, 
    and returns a strictly grounded response dictionary.
    """
```

### Return Signature Specification

- **When grounded evidence exists in PDF:**
  ```python
  {
      "answer": "Peak learning rate: 3e-4 (0.0003), 4,000 linear warmup steps, AdamW optimizer with beta_1 = 0.9 and beta_2 = 0.95 (Section 4.2).",
      "sources": ["page_16_chunk_0"]
  }
  ```
- **When information is absent from PDF (Out-of-Domain):**
  ```python
  {
      "answer": "[not found]",
      "sources": []
  }
  ```

---

## Project 7 Assessment Results

The pipeline was evaluated against all 5 official benchmark questions from Project 7:

| ID | Benchmark Question | Expected Ground Truth | Actual System Output | Source Reference | Evaluation Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1** | What exact learning rate, number of warmup steps, and Adam optimizer beta values were used in the primary pre-training run described in Section 4? | LR: 3e-4 (0.0003), 4,000 linear warmup steps, AdamW beta_1 = 0.9, beta_2 = 0.95 | `Peak learning rate: 3e-4 (0.0003), 4,000 linear warmup steps, AdamW optimizer with beta_1 = 0.9 and beta_2 = 0.95 (Section 4.2).` | `['page_16_chunk_0']` (Page 16) | **PASSED** |
| **Q2** | Who is the current Chief Executive Officer of OpenAI? | Out-of-domain knowledge not present in document. Must reject. | `[not found]` | `[]` | **PASSED** |
| **Q3** | By what percentage did the proposed hybrid retrieval architecture improve Exact Match (EM) accuracy on the benchmark dataset compared with the dense-only baseline? | 14.6% improvement (61.2% to 75.8%) | `The proposed hybrid retrieval architecture improved Exact Match (EM) accuracy by 14.6% (increasing EM accuracy from 61.2% on the dense-only baseline to 75.8%).` | `['page_31_chunk_0']` (Page 31) | **PASSED** |
| **Q4** | What GPU cluster configuration (number and model of GPUs) and total training duration, in days, were required to train the 13-billion-parameter model? | 128 NVIDIA H100 GPUs (80GB SXM5), 400 Gbps InfiniBand, 18.5 days | `Cluster configuration: 128 NVIDIA H100 (80GB SXM5) GPUs interconnected via 400 Gbps InfiniBand; total training duration: 18.5 days.` | `['page_45_chunk_0']` (Page 45) | **PASSED** |
| **Q5** | What was the worldwide box-office revenue of the film Oppenheimer in 2023? | Out-of-domain knowledge not present in document. Must reject. | `[not found]` | `[]` | **PASSED** |

---

## Hallucination Prevention & Rejection Protocol

A critical flaw in naive RAG systems is the tendency to hallucinate answers using internal parametric weights when a query falls outside the knowledge base.

To guarantee compliance:
1. **Zero External Knowledge Injected**: System prompt explicitly blocks parametric recall.
2. **Deterministic Trigger**: The exact sentinel string `[not found]` is enforced.
3. **Empty Source Enforcement**: When `[not found]` triggers, `sources` is guaranteed to return an empty list `[]`, ensuring that unrelated chunks are never falsely attributed.
4. **Tested Guardrails**: Verified on **Q2** (OpenAI leadership) and **Q5** (Oppenheimer box-office statistics), both of which are absent from the technical monograph.

---

## Setup & Execution

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/lokeshsahu2804-korba/mirai-rag-research-assistant.git
cd mirai-rag-research-assistant

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create or edit `.env` in the project root:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Note: `.env` and `chroma_db/` are strictly ignored by `.gitignore` to prevent secret leakage).*

### 5. Run Benchmark Evaluation
```bash
python3 run_questions.py
```

### 6. Using as a Python Module
```python
from rag_pipeline import answer

result = answer("What was the GPU cluster configuration used to train the 13B model?")
print(result["answer"])
# "Cluster configuration: 128 NVIDIA H100 (80GB SXM5) GPUs..."
print(result["sources"])
# ['page_45_chunk_0']
```
