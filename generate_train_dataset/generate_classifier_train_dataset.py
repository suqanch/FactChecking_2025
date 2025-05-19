import json
import csv
import random
import pandas as pd

train_csv_path = "data/dev-embed.csv"      
output_csv_path = "local_data/dev-classifier.csv"

# csv;
# sent0,sent1,hard_neg

# Output: (sent0, sent1), label1
#          (sent0, hard_neg), label0

df = pd.read_csv(train_csv_path, sep=',')

examples = []

for _, row in df.iterrows():
    sent0 = row["sent0"]
    sent1 = row["sent1"]
    hard_neg = row["hard_neg"]

    examples.append({"claim": sent0, "evidence": sent1, "label": 1})  
    examples.append({"claim": sent0, "evidence": hard_neg, "label": 0}) 

random.shuffle(examples)

output_df = pd.DataFrame(examples)
output_df.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

print(f'Transformed {len(output_df)} samples and saved to {output_csv_path}')