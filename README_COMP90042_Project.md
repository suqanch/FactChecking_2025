# COMP90042 Project — Multi-Phase Fact-Checking of Climate Claims  
**Group 24 — Semester 1, 2025**

## All fin-tuned models can be found at:
https://drive.google.com/drive/u/0/folders/1spT1gTUSkm4mP_mTiW0CES2dqKgLPRq_

##  Overview
This project implements a multi-stage pipeline for fact-checking climate-related claims. The system retrieves evidence from a large corpus and classifies claims into four categories: `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`, or `DISPUTED`. The pipeline combines lexical retrieval, semantic dense encoding, transformer-based re-ranking, and zero-shot classification using compact LLMs.

##  Pipeline Architecture

1. **Preprocessing**  
   - Stage-specific tokenization, lemmatization, and stopword removal.
   - Adaptive strategy for each module: retrieval vs. classification.

2. **Hybrid Retrieval**
   - `BM25` lexical retrieval (Pyserini).
   - `MiniLM` bi-encoder (DPR framework) for semantic retrieval.
   - Merge top 100 from each into a hybrid pool.

3. **Cross-Encoder Re-ranking**
   - Re-ranks using a `MiniLM` cross-encoder.
   - Curriculum training: easy → hard negatives.
   - Output: Top-5 evidence per claim.

4. **Zero-Shot Classification**
   - In-context classification using `TinyLlama`, `Qwen1.5`, and `Phi-1.5`.
   - Prompts instruct model to handle `DISPUTED` and `NOT_ENOUGH_INFO` cases explicitly.

##  File Structure

```
COMP90042_Project_2025/
│
├── COMP90042_Project_2025.ipynb      # Main notebook (code + outputs)
├── eval.py                           # Provided evaluation script
├── evidence.json                     # Evidence corpus
├── train-claims.json                 # Labelled training claims
├── dev-claims.json                   # Development claims with labels
├── test-claims-unlabelled.json      # Unlabelled test set for leaderboard
├── README.md                         # This file
└── (Optional scripts if added by you)
```

##  Running the Project

1. **Google Colab Setup**
   - Mount Google Drive and set the correct `data_dir` path.
   - Ensure evidence and claims JSON files are stored under the path.

2. **Install Required Libraries**
   ```python
   !pip install pyserini
   !pip install transformers sentence-transformers faiss-cpu
   !pip install nltk
   ```

3. **Run Notebook Sections in Order**
   - Preprocessing
   - Lexical & dense retrieval
   - Re-ranking via cross-encoder
   - Zero-shot classification
   - Export predictions

4. **Evaluate (Dev Set Only)**
   ```bash
   python eval.py \
       --predictions dev-claims-predictions.json \
       --groundtruth dev-claims.json
   ```

##  Evaluation Metrics

We use three official metrics:

- **Evidence Retrieval F1 (F):** Quality of retrieved evidence passages.
- **Claim Classification Accuracy (A):** Correctness of predicted claim labels.
- **F-A Harmonic Mean:** Final score used for leaderboard and internal analysis.

##  Notes

- Follows all COMP90042 rules: no external APIs, only open-source LLMs under 12GB.
- All models used (MiniLM, TinyLlama) fit in free-tier Colab GPUs.
- SimCSE and BERTopic are implemented as auxiliary methods and evaluated.

##  Contributors

- **Yechen Deng**: Retrieval and cross-encoder re-ranking
- **Zhenyuan He**: SimCSE, in-context learning, classification
- **Wen Zhou**: BERTopic + NER retrieval logic, performance tuning
