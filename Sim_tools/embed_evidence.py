from loguru import logger

import numpy as np
from scipy.stats import spearmanr

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model_Sim import SimcseModel, simcse_unsup_loss, simcse_sup_loss
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

# sample:

# {"evidence-0": "John Bennet Lawes, English entrepreneur and agricultural scientist", 
#  "evidence-1": "Lindberg began his professional career at the age of 16, eventually moving to New York City in 1977.", 
#  "evidence-2": "``Boston (Ladies of Cambridge)'' by Vampire Weekend",

# def embed_evidence(evidence_json_path, output_json_path, model, tokenizer, device,max_length=256):

#     new_evidence_data = {}
#     with open(evidence_json_path, "r", encoding="utf-8") as f:
#         evidence_data = json.load(f)
#         count = 0
#         for evidence_id, evidence_text in evidence_data.items():
#             print(f"正在处理 evidence_id: {evidence_id}，当前进度: {count}/{len(evidence_data)}", evidence_text)
#             count += 1
#             inputs = tokenizer(
#                 evidence_text,
#                 max_length=max_length,
#                 truncation=True,
#                 padding='max_length',
#                 return_tensors='pt'
#             )
#             input_ids = inputs['input_ids'].to(device)
#             attention_mask = inputs['attention_mask'].to(device)
#             token_type_ids = inputs['token_type_ids'].to(device)

#             with torch.no_grad():
#                 embeddings = model(input_ids, attention_mask, token_type_ids)

#             new_evidence_data[evidence_id] = {
#                 "text": evidence_text,
#                 "embedding": embeddings.squeeze(0).tolist()
#             }

#     with open(output_json_path, "w", encoding="utf-8") as f_out:
#         json.dump(new_evidence_data, f_out, ensure_ascii=False, indent=2)



def embed_evidence(evidence_json_path, output_json_path, model, tokenizer, device,max_length=256, batch_size=64):
    df = pd.read_csv(evidence_json_path, sep=',')
    ids = df['id'].tolist()
    evidence = df['text'].tolist()
    model.eval()
    results = {}
    count = 0
    for i in range(0, len(evidence), batch_size):
        batch_evidence = evidence[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        inputs = tokenizer(
            batch_evidence,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)

        with torch.no_grad():
            embeddings = model(input_ids, attention_mask, token_type_ids)

        for j in range(len(batch_evidence)):
            evidence_id = batch_ids[j]
            evidence_text = batch_evidence[j]
            embedding = embeddings[j].cpu().tolist()
            results[evidence_id] = {
                "text": evidence_text,
                "embedding": embedding
            }
            print(f"正在处理 evidence_id: {evidence_id}，当前进度: {count}/{len(evidence)}", evidence_text)
            count += 1
            

    with open(output_json_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)



#dev_file_path json sample:
# {
#     "claim-752": {
#         "claim_text": "[South Australia] has the most expensive electricity in the world.",
#         "claim_label": "SUPPORTS",
#         "evidences": [
#             "evidence-67732",
#             "evidence-572512"
#         ]
#     },

#embedding json sample:
            # results[evidence_id] = {
            #     "text": evidence_text,
            #     "embedding": embedding
            # }



def load_dev_claims(dev_file_path):
    """
    读取 dev 文件，返回一个包含 claim_text、label、evidences 的列表
    """
    claim_x = []
    claim_y = []
    with open(dev_file_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

        for claim_id, claim_info in dev_data.items():
            claim_x.append((claim_id,claim_info["claim_text"]))
            claim_y.append((claim_id, claim_info["evidences"]))
    return claim_x, claim_y

def load_evidence_embeddings(embedding_file_path):
    """
    加载 embedding 文件，兼容 list 或 dict 格式，返回 {evidence_id: {...}} 字典
    """
    embedding_list = []
    with open(embedding_file_path, 'r', encoding='utf-8') as f:
        embedding_data = json.load(f)
        for evidence_id, evidence_info in embedding_data.items():
            embedding = evidence_info.get("embedding", None)
            embedding_list.append((evidence_id, embedding))

    sorted_embeddings = sorted(embedding_list, key=lambda x: x[0])
    embeddings = {embed[1] for embed in sorted_embeddings}
    return embeddings


            

   

if __name__ == '__main__':


    batch_size = 16
    max_length = 256
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    checkpoint = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    model = SimcseModel(pretrained_model=checkpoint, pooling='pooler', dropout=0.1).to(device)
    model.load_state_dict(torch.load("saved_model/best_model.pt", map_location=device))
    model.eval()

        
    evidence_csv_path = "data/evidence.json"  
    output_csv_path = "data/evidence_embed.json" 
    embed_evidence(evidence_csv_path, output_csv_path, model, tokenizer, device, max_length)

    dev_file_path = 'data/dev-claims.json'
