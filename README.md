# CSAI 422 Assignment 4: Advanced RAG with PopQA

## Project Overview

This project implements an advanced Retrieval-Augmented Generation (RAG) pipeline for factual question answering using the PopQA dataset.

The system includes the following components:

- PopQA dataset loading and inspection
- Wikipedia-based retrieval corpus construction
- Dense retrieval using SentenceTransformers and FAISS
- Query expansion
- Hybrid retrieval using BM25 and dense retrieval
- Reciprocal Rank Fusion for combining rankings
- Cross-encoder reranking
- Citation-grounded answer generation
- Self-reflective answer checking
- Retrieval evaluation using Recall@k, Precision@k, and MRR

The goal of the project is to understand how modern RAG systems combine retrieval, ranking, and generation to produce more reliable answers.

---

# Dataset

The dataset used in this project is **PopQA**, which is available from HuggingFace:

https://huggingface.co/datasets/akariasai/PopQA

The dataset contains factual questions along with possible answers and metadata about related Wikipedia entities.

To make the experiments easier to run in Google Colab, a smaller evaluation subset of the dataset was selected.

---

# Main Libraries

The project uses the following Python libraries:

- datasets
- sentence-transformers
- faiss-cpu
- rank_bm25
- wikipedia
- pandas
- numpy
- tqdm
- scikit-learn
- openai

These libraries support dataset loading, embedding generation, retrieval, ranking, and evaluation.

---

# How to Run the Code

1. Open the notebook in **Google Colab**.
2. Run the **installation cell first** to install the required libraries.
3. Run the notebook cells **from top to bottom**.

Start with a small evaluation size:

```python
EVAL_SIZE = 30
```

If you are not using an API key, keep:

```python
USE_OPENAI = False
```

The notebook will automatically generate:

- retrieval examples
- metric comparison tables
- grounded question answering examples
- failure analysis results
- self-reflection examples

---

# Pipeline Steps

## 1. Dataset Preparation

The PopQA dataset is loaded from HuggingFace. The dataset structure is inspected to identify the question field, answer field, and useful metadata.

## 2. Corpus Construction

Wikipedia pages are collected using the subject and object titles provided in the dataset. The pages are cleaned and split into smaller passages. Each passage is stored with a unique identifier.

## 3. Dense Retrieval

The system uses the embedding model:

sentence-transformers/all-MiniLM-L6-v2

The passages are embedded and stored in a FAISS vector index. Questions are embedded and compared with passage embeddings to retrieve the most relevant passages.

## 4. Query Expansion

Query expansion is implemented by adding additional factual keywords to the original question. This helps the retriever capture more relevant information.

## 5. Hybrid Retrieval

Hybrid retrieval combines:

- Dense retrieval (semantic similarity)
- BM25 lexical retrieval (keyword matching)

The results are merged using **Reciprocal Rank Fusion (RRF)**.

## 6. Reranking

A cross-encoder model is used to rerank the retrieved passages:

cross-encoder/ms-marco-MiniLM-L-6-v2

The reranker evaluates each query–passage pair and produces a more accurate ranking.

## 7. Citation-Grounded Generation

The generator produces answers using only the retrieved passages. Each answer includes explicit citations such as:

[P1], [P2]

These citations correspond to the retrieved passages used as evidence.

## 8. Self-Reflection

A reflection stage checks whether the generated answer is:

- grounded in evidence
- properly cited
- complete

If the answer is weak or unsupported, the system revises it using the same evidence.

---

# Evaluation Metrics

The retrievers are evaluated using the following metrics:

**Recall@k**

Measures whether at least one of the top-k retrieved passages contains the correct answer.

**Precision@k**

Measures the proportion of retrieved passages that contain the correct answer.

**Mean Reciprocal Rank (MRR)**

Measures how early the first correct passage appears in the ranking.

The following system configurations are compared:

1. Baseline dense retrieval
2. Dense retrieval with query expansion
3. Hybrid retrieval
4. Hybrid retrieval with reranking
5. Final RAG system with self-reflection

---

# Output Files

The notebook generates several output files:

```
retrieval_comparison_metrics.csv
final_system_comparison.csv
failure_analysis.csv
```

These files contain the evaluation results and analysis produced during the experiments.

---

# Screenshots Included in the Report

The report includes screenshots showing:

1. Dataset structure
2. Corpus and index statistics
3. Dense retrieval examples
4. Baseline metric table
5. Query expansion examples
6. Hybrid retrieval comparison
7. Before and after reranking
8. Citation-grounded QA examples
9. Failure analysis table
10. Self-reflection example
11. Final system comparison table

---

# Notes

The system can run without an API key using the fallback answer generation method. If an OpenAI API key is available, it can be used to improve the answer generation and reflection steps.

---

# Conclusion

This project demonstrates how advanced retrieval techniques can improve the performance of Retrieval-Augmented Generation systems.

Hybrid retrieval combined with reranking and self-reflection produces more reliable and better-supported answers compared to a simple dense retrieval baseline.
