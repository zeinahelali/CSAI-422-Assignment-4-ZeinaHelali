# CSAI 422 Assignment 4: Advanced RAG with PopQA

## Project Overview

This project implements an advanced Retrieval-Augmented Generation (RAG) pipeline for factual question answering using the PopQA dataset.

The system includes:

- PopQA dataset loading
- Wikipedia-based retrieval corpus construction
- Dense retrieval using SentenceTransformers and FAISS
- Query expansion
- Hybrid retrieval using BM25 + dense retrieval
- Reciprocal Rank Fusion
- Cross-encoder reranking
- Citation-grounded answer generation
- Self-reflective answer checking
- Retrieval evaluation using Recall@k, Precision@k, and MRR

## Dataset

The dataset used is PopQA from HuggingFace:

`akariasai/PopQA`

A subset of the dataset was used to make the experiments easier to run in Google Colab.

## Main Libraries

The project uses the following Python libraries:

```bash
datasets
sentence-transformers
faiss-cpu
rank_bm25
wikipedia
pandas
numpy
tqdm
scikit-learn
openai

## How to Run
Open the notebook in Google Colab.
Run the installation cell first.
Run all notebook cells from top to bottom.
Start with a small evaluation size such as:
EVAL_SIZE = 30
If using no API key, keep:
USE_OPENAI = False
The notebook will generate:
retrieval examples
metric comparison tables
grounded QA examples
failure analysis
self-reflection examples
Pipeline Steps
1. Dataset Preparation

The PopQA dataset is loaded from HuggingFace. The question field, possible answer field, and Wikipedia metadata fields are inspected.

2. Corpus Construction

Wikipedia pages are collected using the subject and object Wikipedia titles from PopQA. The pages are cleaned and split into smaller passages. Each passage keeps a unique passage ID.

3. Dense Retrieval

The system uses sentence-transformers/all-MiniLM-L6-v2 to embed passages and questions. FAISS is used to build a vector index for efficient retrieval.

4. Query Expansion

A simple query expansion method adds factual search terms to the original question. This helps test whether additional query context improves retrieval.

5. Hybrid Retrieval

BM25 lexical retrieval is combined with dense retrieval. The final ranking is produced using Reciprocal Rank Fusion.

6. Reranking

A cross-encoder model, cross-encoder/ms-marco-MiniLM-L-6-v2, reranks the retrieved passages to improve the final top-k results.

7. Citation-Grounded Generation

The system generates answers using retrieved passages. Each answer includes passage citations such as [P1] and [P2].

8. Self-Reflection

A reflection stage checks whether the generated answer is grounded, complete, and properly cited. It can revise weak answers using the same retrieved evidence.

Evaluation Metrics

The retrieval systems are evaluated using:

Recall@1, Recall@3, Recall@5
Precision@1, Precision@3, Precision@5
MRR

These metrics compare the baseline dense retriever, query-expanded retriever, hybrid retriever, reranked retriever, and final self-reflective RAG system.

Output Files

The notebook saves the following CSV files:

retrieval_comparison_metrics.csv
final_system_comparison.csv
failure_analysis.csv
Screenshots Included in Report

The report includes screenshots of:

Dataset structure
Corpus and index statistics
Dense retrieval examples
Baseline metric table
Query expansion examples
Hybrid retrieval comparison
Before and after reranking
10 citation-grounded QA examples
Failure analysis table
Self-reflection example
Final system comparison table
Notes

This project can run without an API key by using the built-in fallback answer generation. If an OpenAI API key is available, the same notebook can use it for stronger answer generation and reflection.

Conclusion

The final system shows that hybrid retrieval, reranking, and self-reflection improve the reliability of RAG systems compared to simple dense retrieval. The best configuration is the hybrid retrieval system with reranking and self-reflective answer checking.
