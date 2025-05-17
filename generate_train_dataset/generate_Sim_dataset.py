import json
import csv
import random

train_json_path = "data/train-claims.json"      # claim & evidence 
evidence_json_path = "local_data/train_embed/evidence_32809.json"  # evidence
output_csv_path = "local_data/train_embed/train-embed.csv"     # output csv path
output_csv_path2 = "local_data/train_embed/dev-embed.csv"   
with open(train_json_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(evidence_json_path, "r", encoding="utf-8") as f:
    evidence_data = json.load(f)

all_evidence_ids = list(evidence_data.keys())

triplets = []
count = 0   
for claim_id, claim_info in train_data.items():

    claim_text = claim_info["claim_text"]
    positive_ids = claim_info["evidences"]
    
    for pos_id in positive_ids:
        if pos_id not in evidence_data:
            continue
        pos_text = evidence_data[pos_id]

        negative_candidates = [eid for eid in all_evidence_ids if eid not in positive_ids]
        for _ in range(3):
            neg_id = random.choice(negative_candidates)
            neg_text = evidence_data[neg_id]

            triplets.append((claim_text, pos_text, neg_text))
    # print(f'processing claim_id: {claim_id} {count}/{len(train_data)}', {(claim_text, pos_text, neg_text)})
    count += 1


random.shuffle(triplets)
split_idx = int(len(triplets) * 0.8)
train_triplets = triplets[:split_idx]
dev_triplets = triplets[split_idx:]

with open(output_csv_path, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["sent0", "sent1", "hard_neg"]) 
    writer.writerows(train_triplets)

with open(output_csv_path2, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["sent0", "sent1", "hard_neg"])
    writer.writerows(dev_triplets)

print(f"saved train triplets to {output_csv_path}")
print(f"saved dev triplets to {output_csv_path2}")
print(f'genrate train triplets: {len(train_triplets)}')
print(f'genrate dev triplets: {len(dev_triplets)}')
