import json

# [
#   {
#     "claim_id": "claim-752",
#     "claim_text": "expensive electricity world",
#     "claim_label": "SUPPORTS",
#     "top_evidence_ids": [
#       "evidence-407584",
#       "evidence-164501",
#       "evidence-794314",
#       "evidence-397489",
#       "evidence-1104259"
#     ]
#   },
with open("task4_input_distilbert_from_dev.json", "r", encoding="utf-8") as f:
    dev_claims = json.load(f)  # dict: id -> text

with open("evidence-preprocessed3.json", "r", encoding="utf-8") as f:
    evidence = json.load(f)  # dict: id -> text

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen1.5-1.8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()


# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

# model_name = "microsoft/phi-2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()



# {
#     "claim-752": {
#         "claim_text": "[South Australia] has the most expensive electricity in the world.",
#         "claim_label": "SUPPORTS",
#         "evidences": [
#             "evidence-67732",
#             "evidence-572512"
#         ]
#     },
def load_dev_claims(dev_claims):
    return [
        {
            "claim_text": claim["claim_text"],
            "claim_label": claim["claim_label"],
            "evidences": [evidence for evidence in claim["top_evidence_ids"] if evidence in evidence]
        }
        for claim in dev_claims.values()
    ]

def build_prompt(claim, evidences):
    few_shot = """
You are a fact-checking assistant. Based on the claim and its evidence(s), classify the claim into one of the following categories: {SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED}.
Return format: Label: <label>

If previous evidence contradicts current evidence, classify the claim as DISPUTED.
If all evidence not relevant to the claim, classify the claim as NOT_ENOUGH_INFO.
"""
    ev_lines = "\n".join([f"Evidence {i+1}: {ev}" for i, ev in enumerate(evidences)])
    return few_shot + f"\n\n### Example 4\nClaim: {claim}\n{ev_lines}\nLabel:"

def predict_qwen_causal(claim, evidences):
    prompt = build_prompt(claim, evidences)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    label = decoded.split("Label:")[-1].strip().split()[0].upper()
    return label


match = 0
total = 0
results={}


for claim in load_dev_claims(dev_claims):
    claim_text = claim["claim_text"]
    claim_label = claim["claim_label"]
    evidences = [evidence[eid]["text"] for eid in claim["evidences"]]
    pred_label = predict_qwen_causal(claim_text, evidences)
    if pred_label not in ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]:
        print(f"Invalid label: {pred_label}")
        pred_label = "SUPPORTS"
    results[claim_text] = {
        "claim_text": claim_text,
        "claim_label": pred_label,
        "evidences": evidences
    }
    if pred_label == claim_label:
        match += 1
    total += 1
    # print(f"Claim: {claim_text}\nPredicted: {pred_label}, Actual: {claim_label}\n")

print(f"Accuracy: {match / total:.4f}" if total > 0 else "No claims to evaluate.")

with open("task4_output_Qwen_disBert.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)