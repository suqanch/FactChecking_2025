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
 

def load_train_data_supervised(tokenizer, file_path, max_length = 512):
    feature_list = []
    df = pd.read_csv(file_path, sep=',')      # 读取 CSV，包含三列：sent0、sent1、hard_neg
    rows = df.to_dict('records')
    for row in rows:
        sent0    = row['sent0']      # anchor sentence
        sent1    = row['sent1']      # positive
        hard_neg = row['hard_neg']   # negative
        # tokenizer 批量编码三句话：
        feature = tokenizer(
            [sent0, sent1, hard_neg],
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        # feature 是个 dict，包含：
        # {
        #   'input_ids':      torch.LongTensor of shape [3, seq_len],
        #   'attention_mask': torch.LongTensor of shape [3, seq_len],
        #   'token_type_ids': torch.LongTensor of shape [3, seq_len]
        # }
        feature_list.append(feature)
    return feature_list

def train_sup(model, train_loader, optimizer, device, epochs = 5, train_mode = 'supervise'):
    logger.info("start training")
    model.train()
    best = 0
    accumulation_steps = 4

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
            if train_mode == 'unsupervise':
                loss = simcse_unsup_loss(out, device)
            else:
                loss = simcse_sup_loss(out, device)
        
            loss.backward()
            step += 1

            if (step + 1) % accumulation_steps == 0:
                optimizer.zero_grad()
                optimizer.step()
            
            if step % 100 == 0:
                logger.info(f"epoch: {epoch}, step: {step}, loss: {loss.item()}")

    return 0

# def AMP_train_sup(model, train_loader, optimizer, device, epochs = 5, train_mode = 'supervise'):
#     logger.info("AMP start training")
#     model.train()
#     accumulation_steps = 4
#     scaler = GradScaler()
#     for epoch in range(epochs):
#         for batch_idx, data in enumerate(tqdm(train_loader)):
#             step = epoch * len(train_loader) + batch_idx
#             # [batch, n, seq_len] -> [batch * n, sql_len]
#             sql_len = data['input_ids'].shape[-1]
#             input_ids = data['input_ids'].view(-1, sql_len).to(device)
#             attention_mask = data['attention_mask'].view(-1, sql_len).to(device)
#             token_type_ids = data['token_type_ids'].view(-1, sql_len).to(device)

#             with autocast():
#                 out = model(input_ids, attention_mask, token_type_ids)
#                 if train_mode == 'unsupervise':
#                     loss = simcse_unsup_loss(out, device)
#                 else:
#                     loss = simcse_sup_loss(out, device)

#             scaler.scale(loss).backward()
#             step += 1

#             if (step + 1) % accumulation_steps == 0:
#                 optimizer.zero_grad()
#                 scaler.step(optimizer)
#                 scaler.update()            
#             if step % 100 == 0:
#                 logger.info(f"epoch: {epoch}, step: {step}, loss: {loss.item()}")

#     return 0

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