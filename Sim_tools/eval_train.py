from loguru import logger

import numpy as np
from scipy.stats import spearmanr
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model_Sim import SimcseModel, simcse_sup_loss
from dataset_Sim import TrainDataset, TestDataset
from transformers import BertModel, BertConfig, BertTokenizer
import os
import random
import pandas as pd
import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

import json
import pandas as pd


def match_evidence_by_similarity(claim_embedding, evidence_embeddings_dict, top_k=5, temperature=0.05):

    evidence_ids = list(evidence_embeddings_dict.keys())
    evidence_tensor = torch.stack([evidence_embeddings_dict[eid]['embedding'] for eid in evidence_ids])  # [num_evidence, 768]

    sim_scores = F.cosine_similarity(claim_embedding.unsqueeze(0), evidence_tensor, dim=1)  # [num_evidence]

    sim_probs = F.softmax(sim_scores / temperature, dim=0)  # [num_evidence]
    topk_probs, topk_indices = torch.topk(sim_probs, top_k)
    # print(topk_probs)
    top_evidence_ids = [evidence_ids[i] for i in topk_indices]

    return top_evidence_ids


def varify_evidence(train_json_path, evidence_embeddings_dict, model, tokenizer, device, max_length=256, top_k=5, temperature=0.0001):

    with open(train_json_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    total_claims = 0
    total_hits = 0

    for claim_id, claim_info in train_data.items():
        claim_text = claim_info["claim_text"]
        positive_ids = set(claim_info["evidences"])  # ground truth ids as set

        # Tokenize claim
        inputs = tokenizer(
            claim_text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)

        # Get claim embedding
        with torch.no_grad():
            claim_embedding = model(input_ids, attention_mask, token_type_ids)  # [1, 768]
            claim_embedding = claim_embedding.squeeze(0)  # -> [768]

        # Get top-k matching evidence ids
        result_lst = match_evidence_by_similarity(claim_embedding, evidence_embeddings_dict, top_k=top_k, temperature=temperature)

        # Evaluate hit (if any of top-k is in positive ids)
        hit = any(eid in positive_ids for eid in result_lst)
        total_hits += int(hit)
        total_claims += 1

    accuracy = total_hits / total_claims if total_claims > 0 else 0.0
    print(f"Top-{top_k} Accuracy: {accuracy:.4f} ({total_hits}/{total_claims})")
    return accuracy