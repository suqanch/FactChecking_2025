from torch.utils.data import DataLoader
from model_Sim import ClaimEvidenceClassifier  
from dataset_Sim import ClaimEvidenceEmbeddingDataset
import torch.nn as nn
import torch
import os
from tqdm import tqdm

def train_classifier(
    csv_path,
    encoder,
    tokenizer,
    device,
    epochs=3,
    batch_size=32,
    lr=2e-5,
    save_dir="saved_classifier"
):
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pt")
    final_model_path = os.path.join(save_dir, "final_model.pt")

    dataset = ClaimEvidenceEmbeddingDataset(csv_path, encoder, tokenizer, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ClaimEvidenceClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for claim_emb, evidence_emb, label in dataloader:
            claim_emb = claim_emb.to(device)
            evidence_emb = evidence_emb.to(device)
            label = label.to(device)

            logits = model(claim_emb, evidence_emb)
            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), best_model_path)
                print(f"[Best] Epoch {epoch} | Loss: {loss.item():.4f} | saved to {best_model_path}")

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), final_model_path)
    print(f"[Final] saved {final_model_path}")