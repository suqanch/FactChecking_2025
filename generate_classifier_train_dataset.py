import json
import csv
import random
import pandas as pd

train_csv_path = "data/dev-embed.csv"      
output_csv_path = "local_data/dev-classifier.csv"

# csv格式;
# sent0,sent1,hard_neg

# 输出格式: (sent0, sent1), label1
#          (sent0, hard_neg), label0

df = pd.read_csv(train_csv_path, sep=',')

examples = []

for _, row in df.iterrows():
    sent0 = row["sent0"]
    sent1 = row["sent1"]
    hard_neg = row["hard_neg"]

    examples.append({"claim": sent0, "evidence": sent1, "label": 1})  # 正样本
    examples.append({"claim": sent0, "evidence": hard_neg, "label": 0})  # 负样本

# 打乱顺序
random.shuffle(examples)

# 保存为新的 CSV
output_df = pd.DataFrame(examples)
output_df.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

print(f"转换完成，共生成样本数: {len(output_df)}，保存至: {output_csv_path}")