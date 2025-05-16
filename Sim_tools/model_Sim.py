import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, BertTokenizer
# from utils.modeling import BertModel as SimBertModel
# from utils.modeling import BertConfig as SimBertConfig
from transformers import BertTokenizer



class ClaimEvidenceClassifier(nn.Module):
    def __init__(self, input_size = 768, hidden_size=256, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)  
        )

    def forward(self, claim_emb, evidence_emb):
        """
        claim_inputs: dict with input_ids, attention_mask, token_type_ids (BERT style)
        evidence_inputs: same
        """

        # cat [claim, evidence, |claim - evidence|]
        x = torch.cat([claim_emb, evidence_emb, torch.abs(claim_emb - evidence_emb)], dim=1)

        logits = self.classifier(x)  # [batch, num_labels]
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits
    


class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""

    def __init__(self, pretrained_model, pooling, dropout=0.3):
        super(SimcseModel, self).__init__()
        # config = SimBertConfig.from_pretrained(pretrained_model)
        config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        # self.bert = SimBertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True, return_dict=True)
        # return out[1]
        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]
        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]




# def simcse_sup_loss(y_pred, device, lamda=0.05):
#     similarities = F.cosine_similarity(y_pred.unsqueeze(0), y_pred.unsqueeze(1), dim=2)
#     row = torch.arange(0, y_pred.shape[0], 3)
#     col = torch.arange(0, y_pred.shape[0])
#     col = col[col % 3 != 0]

#     similarities = similarities[row, :]
#     similarities = similarities[:, col]
#     similarities = similarities / lamda

#     y_true = torch.arange(0, len(col), 2, device=device)
#     loss = F.cross_entropy(similarities, y_true)
#     return loss


def simcse_sup_loss(y_pred, device, lamda=0.05):
     y_true = torch.arange(y_pred.shape[0], device=device)
     use_row = torch.where((y_true + 1) % 3 != 0)[0]
     y_true = (use_row - use_row % 3 * 2) + 1
     sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
     sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
     sim = torch.index_select(sim, 0, use_row)
     sim = sim / lamda
     loss = F.cross_entropy(sim, y_true)
     return loss

if __name__ == '__main__':
    y_pred = torch.rand((30 ,16))
    loss = simcse_sup_loss(y_pred, 'cpu', lamda=0.05)
    print(loss)
