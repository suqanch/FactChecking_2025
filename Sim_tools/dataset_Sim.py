import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    



class ClaimEvidenceEmbeddingDataset(Dataset):
    def __init__(self, csv_path, encoder_model, tokenizer, device, max_length=256):
        self.data = pd.read_csv(csv_path)
        self.encoder = encoder_model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def encode_text(self, text):
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        token_type_ids = encoded['token_type_ids'].to(self.device)

        with torch.no_grad():
            embedding = self.encoder(input_ids, attention_mask, token_type_ids)  # [1, 768]

        return embedding.squeeze(0)  # [768]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        claim_emb = self.encode_text(row['claim'])
        evidence_emb = self.encode_text(row['evidence'])
        label = torch.tensor(row['label'], dtype=torch.float)  # 如果是 BCE loss，保持 float

        return claim_emb, evidence_emb, label