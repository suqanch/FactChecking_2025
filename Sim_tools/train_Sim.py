from loguru import logger

import numpy as np
from scipy.stats import spearmanr

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import json
from embed_evidence import embed_evidence, embed_evidence_pkl, load_evidence_embeddings_from_pickle
from model_Sim import SimcseModel, simcse_sup_loss
from dataset_Sim import TrainDataset, TestDataset
from transformers import BertModel, BertConfig, BertTokenizer
import os
import random
import pandas as pd
import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
 

def load_train_data_supervised(tokenizer, file_path, max_length = 128):
    feature_list = []
    df = pd.read_csv(file_path, sep=',')      # read CSV：sent0、sent1、hard_neg
    rows = df.to_dict('records')
    for row in rows:
        sent0    = row['sent0']      # anchor sentence
        sent1    = row['sent1']      # positive
        hard_neg = row['hard_neg']   # negative
        
        feature = tokenizer(
            [sent0, sent1, hard_neg],
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        # feature is a dict with keys:
        # {
        #   'input_ids':      torch.LongTensor of shape [3, seq_len],
        #   'attention_mask': torch.LongTensor of shape [3, seq_len],
        #   'token_type_ids': torch.LongTensor of shape [3, seq_len]
        # }
        feature_list.append(feature)
    return feature_list

def load_test_data_supervised(tokenizer, file_path, max_length = 128):
    feature_list = []
    df = pd.read_csv(file_path, sep=',')      # read CSV: sent0、sent1、hard_neg
    rows = df.to_dict('records')
    for row in rows:
        sent0    = row['sent0']      # anchor sentence
        sent1    = row['sent1']      # positive
        hard_neg = row['hard_neg']   # negative
        
        feature = tokenizer(
            [sent0, sent1, hard_neg],
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        feature_list.append(feature)
    return feature_list


def train_sup(model, train_loader, dev_loader, optimizer, device, tokenizer, epochs = 10, eval_step = 400, 
              train_mode = 'supervise', dev_claim_path = 'dev-claims.json', 
              evidence_subset_path = 'evidence_subset_train.json', 
              output_pkl_path = 'evidence_embeddings.pkl'):
    logger.info("start training")

    train_loss_log = []
    eval_loss_log  = []
    accuracy_log   = []
    f1_log = []

    model.train()
    accumulation_steps = 2
    best_loss = float("inf")
    best_accuracy = 0.0
    save_path = "saved_model"
    os.makedirs(save_path, exist_ok=True)


    dev_accuracy, f1_score = eval_accuracy(model, dev_claim_path, evidence_subset_path, output_pkl_path, tokenizer, device)
    logger.info(f"Step 0 retrieval_accuracy: {dev_accuracy:.4f}")
    logger.info(f"Step 0 f1_score: {f1_score:.4f}")

    for epoch in range(epochs):
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx
            # [batch, n, seq_len] -> [batch * n, sql_len]
            sql_len = data['input_ids'].shape[-1]
            input_ids = data['input_ids'].view(-1, sql_len).to(device)
            attention_mask = data['attention_mask'].view(-1, sql_len).to(device)
            token_type_ids = data['token_type_ids'].view(-1, sql_len).to(device)
            # logger.info('debug')
            out = model(input_ids, attention_mask, token_type_ids)

            loss = simcse_sup_loss(out, device)

            loss = loss / accumulation_steps
            loss.backward()
            if (step + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):   
                logger.info(f"epoch: {epoch}, step: {step}, loss: {loss.item()}")
                train_loss_log.append((step, loss.item()))
                optimizer.step()
                optimizer.zero_grad()
            
            # if loss.item() < best_loss:
            #     best_loss = loss.item()
            #     torch.save(model.state_dict(), os.path.join(save_path, "best_model.pt"))
            #     logger.info(f"Best model saved at step {step} with loss {best_loss:.4f}")

            if step % eval_step == 0:
                eval_loss = bt_eval_loss(model, dev_loader, device)
                eval_loss_log.append((step, eval_loss))
                logger.info(f"epoch: {epoch}, step: {step}, eval_loss: {eval_loss:.4f}")
                model.train()
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    torch.save(model.state_dict(), os.path.join(save_path, "best_model_bt.pt"))
                    logger.info(f"Best model saved at step {step} with loss {best_loss:.4f}")

        dev_accuracy, f1_score = eval_accuracy(model, dev_claim_path, evidence_subset_path, output_pkl_path, tokenizer, device)
        accuracy_log.append((step, dev_accuracy))
        f1_log.append((step, f1_score))
        logger.info(f"epoch: {epoch}, step: {step}, retrieval_accuracy: {dev_accuracy:.4f}")
        logger.info(f"epoch: {epoch}, step: {step}, f1_score: {f1_score:.4f}")
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pt"))
            logger.info(f"Best model saved at step {step} with accuracy {best_accuracy:.4f}")
        model.train()
        # dev_accuracy = eval_accuracy(model, dev_claim_path, evidence_subset_path, output_pkl_path, tokenizer, device)
        # accuracy_log.append((step, dev_accuracy))
        # logger.info(f"epoch: {epoch}, step: {step}, retrieval_accuracy: {dev_accuracy:.4f}")

    logger.info(f"Training completed. Best loss: {best_loss:.4f}")
    torch.save(model.state_dict(), os.path.join(save_path, "final_model.pt"))
    return train_loss_log, eval_loss_log, accuracy_log, f1_log


# import pandas as pd

# df_train = pd.DataFrame(train_loss_log, columns=['step', 'train_loss'])
# df_eval  = pd.DataFrame(eval_loss_log, columns=['step', 'eval_loss'])
# df_acc   = pd.DataFrame(accuracy_log, columns=['step', 'accuracy'])
# df_f1    = pd.DataFrame(f1_log, columns=['step', 'f1_score'])

# log_df = pd.merge(df_train, df_eval, on='step', how='outer')
# log_df = pd.merge(log_df, df_acc, on='step', how='outer')
# log_df = pd.merge(log_df, df_f1, on='step', how='outer')
# log_df.sort_values(by='step', inplace=True)

# log_df.to_csv("logs/training_log.csv", index=False)








# dev_claim sample
# {
#     "claim-752": {
#         "claim_text": "[South Australia] has the most expensive electricity in the world.",
#         "claim_label": "SUPPORTS",
#         "evidences": [
#             "evidence-67732",
#             "evidence-572512"
#         ]
#     },
# }


# evidence_subset sample
# {
#   "evidence-442946": "At very high concentrations (100 times atmosphe...
# }

def eval_accuracy(model, dev_claim_path, evidence_subset_path, output_pkl_path, tokenizer, device, top_k = 10):
    model.eval()
    with open(dev_claim_path, "r", encoding="utf-8") as f:
        dev_claim = json.load(f)

    matched_evidence = 0
    correct_evidence = 0
    retrieved_total = 0
    count_5 = 0
    total_count = 0

    embed_evidence_pkl(evidence_subset_path, output_pkl_path, model, tokenizer, device, max_length=256, batch_size=512)

    evidence_embeddings_dict = load_evidence_embeddings_from_pickle(output_pkl_path, device)
    logger.info(f"Load evidence embeddings from {output_pkl_path}")
    evidence_ids = list(evidence_embeddings_dict.keys())
    evidence_embeddings = torch.stack([evidence_embeddings_dict[eid]['embedding'] for eid in evidence_ids])  # [num_evidence, 768]

    for claim_id, claim_info in dev_claim.items():
        claim_text = claim_info["claim_text"]
        positive_ids = claim_info["evidences"]
        # Tokenize claim

        inputs = tokenizer(
            claim_text,
            max_length=256,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        with torch.no_grad():
            claim_embedding = model(input_ids, attention_mask, token_type_ids)

        # calculate cosine similarity for claim and all evidence
        sim_scores = F.cosine_similarity(claim_embedding, evidence_embeddings, dim=1)

        #sort top k evidence
        topk_probs, topk_indices = torch.topk(sim_scores, top_k)
        output_evidence_id = [evidence_ids[i] for i in topk_indices]

        count = (sim_scores > 0.5).sum().item()
        count_5 += count
        total_count += sim_scores.size(0)
        # print(f"Number of evidence with sim > 0.5: {count} / {sim_scores.size(0)}")

        matched = set(output_evidence_id) & set(positive_ids)
        matched_evidence += len(matched)        
        correct_evidence += len(positive_ids)
        retrieved_total += len(output_evidence_id)
    
    recall = matched_evidence / correct_evidence if correct_evidence > 0 else 0.0
    precision = matched_evidence / retrieved_total if retrieved_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    logger.info(f"[Eval Accuracy] Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")
    logger.info(f'[Eval Accuracy] Average number of evidence with sim > 0.5: {count_5 / total_count:.4f}')
    return recall, f1


#Bradley-Terry Loss
def bt_eval_loss(model, dev_loader, device):
    losses = []
    model.eval()
    # model.to(device)  

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dev_loader)):

            # [batch, n, seq_len] -> [batch * n, sql_len]
            sql_len = data['input_ids'].shape[-1]
            input_ids = data['input_ids'].view(-1, sql_len).to(device)
            attention_mask = data['attention_mask'].view(-1, sql_len).to(device)
            token_type_ids = data['token_type_ids'].view(-1, sql_len).to(device)

            # Get model output for all (anchor, pos, neg) in batch
            out = model(input_ids, attention_mask, token_type_ids)  # shape: [B*3, hidden_dim]
            # Split embeddings into anchor, pos, neg
            e_a = out[0::3]  # from first one, every 3rd → a1, a2, ...
            e_p = out[1::3]  # from first one, every 3rd → p1, p2, ...
            e_n = out[2::3]  # from second one, every 3rd → n1, n2, ...
            # Compute similarity
            sim_ap = F.cosine_similarity(e_a, e_p, dim=1)  # [B]
            sim_an = F.cosine_similarity(e_a, e_n, dim=1)  # [B]

            logits = torch.stack([sim_ap, sim_an], dim=1)  # shape: [B, 2]
            log_probs = F.log_softmax(logits, dim=1)
            loss = -log_probs[:, 0]  
            losses.append(loss.mean().item())

    return sum(losses) / len(losses)

# def bt_eval_loss(model, dataloader, device):
#     losses = []
#     model.eval()
#     with torch.no_grad():
#         for a, p, n in dataloader:
#             e_a = model(a)
#             e_p = model(p)
#             e_n = model(n)

#             sim_ap = F.cosine_similarity(e_a, e_p)
#             sim_an = F.cosine_similarity(e_a, e_n)

#             logits = torch.stack([sim_ap, sim_an], dim=1)
#             log_probs = F.log_softmax(logits, dim=1)
#             loss = -log_probs[:, 0]  # choose the positive pair
#             losses.append(loss.mean().item())

#     return sum(losses) / len(losses)



if __name__ == '__main__':
    batch_size = 64
    file_path = 'data/train-embed.csv'
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    checkpoint = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    model = SimcseModel(pretrained_model=checkpoint, pooling='pooler', dropout=0.1).to(device)
    train_data = load_train_data_supervised(tokenizer, file_path)
    train_dataset = TrainDataset(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-5)
    train_sup(model, train_dataloader, optimizer, device, epochs = 5, train_mode = 'supervise')