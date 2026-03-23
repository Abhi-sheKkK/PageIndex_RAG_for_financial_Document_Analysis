# PageIndex: Reasoning-Based RAG Pipeline

This directory contains a complete, isolated implementation of a **Reasoning-Based RAG** system using the `pageindex` tree-structure instead of traditional vector-based chunking.

---

## 🚀 Key Differences from Traditional RAG

| Feature | Traditional RAG (`src/`) | PageIndex RAG (`src_pageindex/`) |
| :--- | :--- | :--- |
| **Indexing** | Fix-sized Chunks & Vector Embeddings | Hierarchical Document Tree |
| **Retrieval** | Similarity Search (K-nearest neighbors) | **LLM Reasoning Search** over tree nodes |
| **Context** | Individual Text Chunks | Full Semantic Sections/Nodes |
| **Precision** | Risk of loss-of-context in small chunks | High (Preserves document hierarchy) |

---

## 🛠 Project Structure

- **`main.py`**: The central orchestrator. Supports `--step extract`, `--step evaluate`, and `--step all`.
- **`rag_pipeline.py`**: Standardized on **GPT-4o-mini** for both Reasoning Retrieval and Structured Extraction.
- **`pageindex_client.py`**: Wrapper for `PageIndexClient`. Handles tree generation and local caching.
- **`evaluator.py`**: LLM-as-Judge evaluation using GPT-4o-mini.
- **`visualize.py`**: Generates `heatmap_pageindex.png` with automated numeric score handling.
- **`config.py`**: Standardized configuration for OpenAI and PageIndex API keys.

---

## 🔑 Configuration

The pipeline is standardized on **OpenAI** (`gpt-4o-mini`) for all intelligence tasks.

### Required Environment Variables:
Set these in your `.env` file or export them to your shell:
- `OPENAI_API_KEY`: Your standard OpenAI API key.
- `PAGEINDEX_API_KEY`: Your PageIndex API key (used for all documents).

---

## 📦 Features & Caching

### 1. Document Tree Caching
Document structures are expensive to generate. Once a PDF is processed by PageIndex, its tree structure is saved locally in `pageindex_cache/`. Subsequent runs for the same PDF will load from disk, incurring **zero** additional PageIndex costs.

### 2. Evaluation Caching
LLM-as-Judge scores are stored in `pageindex_evaluation_results.csv`. Valid results are never re-evaluated, saving tokens and time. 

### 3. Case-Insensitive Normalization
Automated normalization handles differences between LLM extraction (e.g., `AMAZON.COM, INC.`) and ground truth data (e.g., `Amazon.com, Inc.`).

---

## 🏃 How to Run

1. **Install dependencies**:
   ```bash
   pip install pageindex openai pandas matplotlib seaborn python-dotenv tenacity
   ```

2. **Execute the pipeline**:
   ```bash
   python -m src_pageindex.main --step all
   ```
