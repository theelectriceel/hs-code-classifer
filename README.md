# AI-Powered Harmonized Tariff Classification

### AI for Ireland Hackathon

Developed for the **AI for Ireland Hackathon**, this project implements an intelligent **Harmonized System (HS) tariff code classification pipeline** that combines dense vector retrieval with **LLM-based semantic reasoning** to map product descriptions to their most contextually appropriate tariff classifications.

### Architecture

The system uses a two-stage retrieval and reasoning pipeline:

1. **Semantic Candidate Retrieval** — Product descriptions are transformed into vector embeddings and compared against the HS tariff corpus using semantic similarity search.

2. **Top-K Candidate Selection** — The five most semantically relevant HS codes are retrieved, reducing the classification search space.

3. **LLM Semantic Reranking** — An LLM evaluates the retrieved candidates against the original product description, resolving semantic ambiguity and identifying the most contextually appropriate classification.

4. **Final Classification** — The highest-confidence candidate is returned as the predicted HS tariff code.

### Pipeline

`Product Description → Embedding → Vector Similarity Search → Top-5 Candidates → LLM Semantic Reranking → HS Code`

### Key Features

- Hybrid **embedding retrieval + LLM reasoning** architecture
- Semantic search across Harmonized System tariff descriptions
- Top-k candidate retrieval for efficient classification
- Context-aware tariff code disambiguation
- Designed and developed as part of the **AI for Ireland Hackathon**
