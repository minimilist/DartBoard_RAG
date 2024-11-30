# Dartboard Pipeline Documentation

This repository provides a pipeline for efficient text retrieval and ranking using embeddings and cross-encoders. The system leverages caching for optimized performance and offers several methods for similarity computations, nearest-neighbor searches, and hybrid ranking.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Setup](#setup)
4. [Pipeline Details](#pipeline-details)
    - [Caching](#caching)
    - [Encoding](#encoding)
    - [Distance and Similarity Calculations](#distance-and-similarity-calculations)
    - [Dartboard Algorithms (DART)](#dartboard-algorithms-dart)
    - [Hybrid Retrieval](#hybrid-retrieval)
    - [Maximal Marginal Relevance (MMR)](#maximal-marginal-relevance-mmr)
    - [KNN with Cross-Encoder](#knn-with-cross-encoder)
5. [Key Functions](#key-functions)
6. [Future Improvements](#future-improvements)

---

## Overview

The `dartboard.py` pipeline provides tools for:
- Efficient text embedding and similarity computation.
- Cross-encoder-based reranking for improved relevance.
- Hybrid algorithms combining embeddings and cross-encoder scores.
- Optimized caching to minimize redundant computations.

### Applications
- Document retrieval
- Sentence ranking
- Text clustering

---

## Features

- **Caching:** Persistent storage using `diskcache` to speed up repeated operations.
- **Cross-Encoders:** Fine-grained scoring of text pairs using `sentence-transformers`.
- **Distance Metrics:** Support for cosine similarity and log-normalized distances.
- **Retrieval Methods:** Implements KNN, DART, hybrid, and MMR algorithms.
- **Customizability:** Easily adaptable to different embedding models or cross-encoder variants.

---

## Setup

### Prerequisites
- Python 3.8+
- Required libraries:
  ```bash
  pip install numpy itertools sentence-transformers diskcache scipy scikit-learn
  ```

### Directory Structure
- **Cache Directories:**
  - `./cache/cache_embs/cache_encoder`: Stores embedding results.
  - `./cache/cache_crosscoder`: Stores cross-encoder scores.

### Running the Pipeline
Ensure the required cache directories are present, or they will be created on the first run.

---

## Pipeline Details

### Caching
- Uses `diskcache` for efficient memoization of:
  - Sentence embeddings (`encode` function).
  - Cross-encoder distances (`cc_cache_hack` function).
- Cache eviction policy: `none` (entries persist indefinitely).

### Encoding
- Embedding Model: `multi-qa-mpnet-base-cos-v1` from `sentence-transformers`.
- Cross-Encoder Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Functions:
  - `encode`: Generates embeddings for a given string.
  - `get_crosscoder_dists`: Computes cross-encoder distances for text pairs.

### Distance and Similarity Calculations
- **Cosine Similarity:**
  - `cos_sim`: Computes dot-product-based similarity.
  - `scaled_cosdist`: Normalizes cosine similarity to a distance metric.
- **Log-Normalized Distances:**
  - `lognorm`: Converts distances to probabilities using a log-normal distribution.

### Dartboard Algorithms (DART)
- Greedy selection of top candidates using probabilistic scores.
- Functions:
  - `get_dartboard_crosscoder2`: Uses cross-encoder scores for DART.
  - `get_dartboard_hybrid`: Combines embeddings and cross-encoder for DART.

### Hybrid Retrieval
- Combines:
  - KNN-based initial selection from embeddings.
  - Fine-grained scoring and reranking with a cross-encoder.
- Function: `get_dists_hybrid`.

### Maximal Marginal Relevance (MMR)
- Balances relevance (to query) and diversity (pairwise similarity).
- Function: `get_mmr_crosscoder2`.

### KNN with Cross-Encoder
- Performs KNN on embeddings followed by reranking with a cross-encoder.
- Function: `get_knn_crosscoder`.

---

## Key Functions

### 1. **Encoding and Caching**
```python
@cache2.memoize()
def encode(dstr: str):
    if encode.encoder is None:
        encode.encoder = st.SentenceTransformer('multi-qa-mpnet-base-cos-v1')
    return encode.encoder.encode(dstr)
```

### 2. **Cross-Encoder Distances**
```python
def get_crosscoder_dists(pairs):
    # Computes distances for text pairs using cross-encoder
```

### 3. **DART Algorithms**
```python
def get_dartboard_crosscoder2(get_dists_results, sigma: float, k: int):
    # Implements DART using cross-encoder distances
```

### 4. **Hybrid Retrieval**
```python
def get_dists_hybrid(query, embs, myencode, texts, triage):
    # Combines embedding-based KNN and cross-encoder reranking
```

### 5. **MMR**
```python
def get_mmr_crosscoder2(get_dists_results, diversity: float, k: int):
    # Selects sentences balancing relevance and diversity
```

---

## Future Improvements

1. **Code Refactoring:**
   - Simplify and consolidate redundant code across functions.
   - Modularize common operations for better reusability.

2. **Documentation:**
   - Add docstrings for all functions.
   - Include examples and use cases.

3. **Performance Optimization:**
   - Profile the pipeline to identify bottlenecks.
   - Explore more efficient caching strategies.

4. **Model Variants:**
   - Support for additional embedding and cross-encoder models.

---

## License
This project is licensed under the MIT License.

---

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

---

## Contact
For questions or support, please contact [your email or GitHub profile].
