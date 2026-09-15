# AI-Powered Harmonized Tariff Classification

### Overview

An intelligent **Harmonized System (HS) tariff code classification pipeline** that combines dense vector retrieval with **LLM-based semantic reasoning** to map product descriptions to their most contextually appropriate tariff classifications.

### Architecture

The classification pipeline follows a two-stage retrieval and reasoning architecture:

1. **Semantic Candidate Retrieval** — Product descriptions are transformed into vector embeddings and compared against the HS tariff corpus using semantic similarity search.

2. **Top-K Candidate Selection** — The five highest-ranking HS codes are retrieved, significantly reducing the classification search space.

3. **LLM Semantic Reranking** — The retrieved candidates are passed to an LLM reasoning layer, which evaluates contextual and semantic compatibility between the product description and candidate tariff definitions.

4. **Final Classification** — The model selects the most contextually appropriate HS code from the candidate set.

### Pipeline

`Product Description → Embedding → Vector Similarity Search → Top-5 HS Candidates → LLM Semantic Reranking → Final HS Code`

### Key Features

- Hybrid **embedding + LLM** classification architecture
- Semantic retrieval over Harmonized System tariff descriptions
- Top-k candidate generation for efficient search-space reduction
- Context-aware LLM reranking and disambiguation
- Modular pipeline designed for integration into customs and trade-classification workflows
