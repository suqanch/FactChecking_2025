import json
import csv
import random

train_json_path = "data/train-claims.json"      # 包含claim和关联evidence的结构
dev_json_path = "data/dev-claims.json"      # 包含claim和关联evidence的结构
evidence_json_path = "data/evidence.json"  # 所有evidence的内容
output_evidence_set_path = "local_data/evidence1.json"     
# output_dev_emb_path = "local_data/dev-embed-1.json"  

with open(train_json_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open(dev_json_path, "r", encoding="utf-8") as f:
    dev_data = json.load(f)
with open(evidence_json_path, "r", encoding="utf-8") as f:
    evidence_data = json.load(f)

#combine train and dev data
merged_data = {**train_data, **dev_data}

evicence_set = {}
for claim_id, claim_info in merged_data.items():

    claim_text = claim_info["claim_text"]
    positive_ids = claim_info["evidences"]
    
    for pos_id in positive_ids:
        if pos_id not in evidence_data:
            continue
        evicence_set[pos_id] = evidence_data[pos_id]

# Save the evidence set to a JSON file
with open(output_evidence_set_path, "w", encoding="utf-8") as f:
    json.dump(evicence_set, f, ensure_ascii=False, indent=2)


