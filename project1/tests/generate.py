import pandas as pd
import numpy as np
import os
from pandas.api.types import is_numeric_dtype


# INPUT FILE
input_file = "iris.csv"

# OUTPUT FOLDER
output_dir = "generated_datasets"
os.makedirs(output_dir, exist_ok=True)

# LOAD DATA
df = pd.read_csv(input_file)

# Identify label column (last column)
label_col = df.columns[-1]
feature_cols = df.columns[:-1]

# 1. Original copy
df.to_csv(f"{output_dir}/iris_original.csv", index=False)

# 2. Rename columns
df_renamed = df.copy()
new_cols = [f"feature_{i}" for i in range(len(feature_cols))] + ["label"]
df_renamed.columns = new_cols
df_renamed.to_csv(f"{output_dir}/iris_renamed.csv", index=False)

# 3. Shuffle columns
df_shuffled = df.sample(frac=1, axis=1)
df_shuffled.to_csv(f"{output_dir}/iris_shuffled_columns.csv", index=False)

# 4. Add ID column
df_id = df.copy()
df_id.insert(0, "ID", range(1, len(df_id) + 1))
df_id.to_csv(f"{output_dir}/iris_with_id.csv", index=False)

# 5. Missing values (10%)
df_missing = df.copy()
for col in feature_cols:
    df_missing.loc[df_missing.sample(frac=0.1).index, col] = np.nan
df_missing.to_csv(f"{output_dir}/iris_missing.csv", index=False)

# 6. Binary classification (remove one class)
df_binary = df[df[label_col] != df[label_col].unique()[-1]]
df_binary.to_csv(f"{output_dir}/iris_binary.csv", index=False)

# 7. Regression version (convert label to numeric if needed)
df_reg = df.copy()
if not is_numeric_dtype(df_reg[label_col]):
    unique_vals = {v: i for i, v in enumerate(df_reg[label_col].unique())}
    df_reg[label_col] = df_reg[label_col].map(unique_vals)

df_reg.to_csv(f"{output_dir}/iris_regression.csv", index=False)

# 8. Add noise feature
df_noise = df.copy()
df_noise["random_noise"] = np.random.rand(len(df_noise))
df_noise.to_csv(f"{output_dir}/iris_with_noise.csv", index=False)

# 9. Single feature only
df_single = df[[feature_cols[0], label_col]]
df_single.to_csv(f"{output_dir}/iris_single_feature.csv", index=False)

print("All datasets generated in:", output_dir)