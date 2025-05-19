import json
import csv
import random

train_json_path = "data/train-claims.json"      # claim & evidence
dev_json_path = "data/dev-claims.json"      # claim & evidence
test_json_path = "data/test-claims-unlabelled.json"  # claim
evidence_json_path = "data/evidence.json"  # evidence
output_evidence_set_path = "local_data/evidence_subset_train.json"     
output_claim_set_path = "local_data/claims.json"

# output_dev_emb_path = "local_data/dev-embed-1.json"  

with open(train_json_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open(dev_json_path, "r", encoding="utf-8") as f:
    dev_data = json.load(f)
with open(test_json_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)
with open(evidence_json_path, "r", encoding="utf-8") as f:
    evidence_data = json.load(f)

#combine train and dev data
# merged_data = {**train_data, **dev_data}
merged_data = train_data

evicence_set = {}
for claim_id, claim_info in merged_data.items():

    claim_text = claim_info["claim_text"]
    positive_ids = claim_info["evidences"]
    
    for pos_id in positive_ids:
        if pos_id not in evidence_data:
            print(f"Warning: Evidence ID {pos_id} not found in evidence data.")
            continue
        evicence_set[pos_id] = evidence_data[pos_id]

print(len(evicence_set))
# Save the evidence set to a JSON file
with open(output_evidence_set_path, "w", encoding="utf-8") as f:
    json.dump(evicence_set, f, ensure_ascii=False, indent=4)


claim_set = {}

def get_claims(data):

    claim_set = {}
    for claim_id, claim_info in data.items():
        claim_text = claim_info["claim_text"]
        claim_set[claim_id] = {
            "claim_text": claim_text,
        }
    return claim_set

train_claim_set = get_claims(train_data)
dev_claim_set = get_claims(dev_data)
test_claim_set = get_claims(test_data)
claim_set = {**train_claim_set, **dev_claim_set, **test_claim_set}
print(len(claim_set))
# Save the claim set to a JSON file
with open(output_claim_set_path, "w", encoding="utf-8") as f:
    json.dump(claim_set, f, ensure_ascii=False, indent=4)