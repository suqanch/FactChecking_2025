import json
import csv
import random

# ====== 设置路径 ======
train_json_path = "data/dev-claims.json"      # 包含claim和关联evidence的结构
evidence_json_path = "data/evidence.json"  # 所有evidence的内容
output_csv_path = "data/dev-embed.csv"     # 输出的csv路径

# ====== 加载数据 ======
with open(train_json_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(evidence_json_path, "r", encoding="utf-8") as f:
    evidence_data = json.load(f)

# 所有 evidence ID 列表（用于采样负例）
all_evidence_ids = list(evidence_data.keys())

# ====== 构造三元组样本 ======
triplets = []
count = 0   
for claim_id, claim_info in train_data.items():

    claim_text = claim_info["claim_text"]
    positive_ids = claim_info["evidences"]
    
    for pos_id in positive_ids:
        if pos_id not in evidence_data:
            continue
        pos_text = evidence_data[pos_id]

        # 随机选择一个非当前 positive 的 negative
        negative_candidates = [eid for eid in all_evidence_ids if eid not in positive_ids]
        neg_id = random.choice(negative_candidates)
        neg_text = evidence_data[neg_id]

        triplets.append((claim_text, pos_text, neg_text))
    print(f"正在处理 claim_id: {claim_id}，当前进度: {count}/{len(train_data)}", {(claim_text, pos_text, neg_text)})
    count += 1
# ====== 保存为 CSV 文件 ======
with open(output_csv_path, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["anchor", "positive", "negative"])  # 写表头
    writer.writerows(triplets)

print(f"✅ 成功生成 triplets，保存为: {output_csv_path}")