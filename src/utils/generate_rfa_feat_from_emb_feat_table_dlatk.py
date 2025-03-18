from argparse import ArgumentParser
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import MySQLdb
import sys
import os
from tqdm import tqdm
from sklearn.decomposition import PCA

# Expand the path to ~/.my.cnf
config_file = os.path.expanduser("~/.my.cnf")

parser = ArgumentParser(description="generate embeddings for DLATK")

parser.add_argument("--database_name", type=str, help="Database to create feat table")
parser.add_argument("--input_table_name", type=str, help="Table to create to store feat values")
parser.add_argument("--output_table_name", type=str, help="Table to create to store residualized LDA feat values")
parser.add_argument("--dimensionality_reduction_method", choices=['pca', 'model'], type=str, help="Dimensionality reduction technique/method")
parser.add_argument("--rfa_output_dimension", type=int, help="Residualized Factor Adaptation Output Dimensions. eg 3")

parser.add_argument("--input_group_id_col_name", type=str, default="group_id", help="Input column name: Group ID")
parser.add_argument("--input_feat_col_name", type=str, default="feat", help="Input column name: Feature")
parser.add_argument("--input_value_col_name", type=str, default="value", help="Input column name: Value")
parser.add_argument("--input_group_norm_col_name", type=str, default="group_norm", help="Input column name: Group Norm")

parser.add_argument("--output_group_id_col_name", type=str, default="group_id", help="Output column name: Group ID")
parser.add_argument("--output_feat_col_name", type=str, default="feat", help="Output column name: Feature")
parser.add_argument("--output_value_col_name", type=str, default="value", help="Output column name: Value")
parser.add_argument("--output_group_norm_col_name", type=str, default="group_norm", help="Output column name: Group Norm")

args = parser.parse_args()
print(vars(args))

conn = MySQLdb.connect(
    db=args.database_name, 
    read_default_file=config_file, 
    charset="utf8mb4", 
    use_unicode=True
)
cur = conn.cursor()
import pdb; pdb.set_trace()
# ------------------------------------------------------------------------------
# 1. Drop Output Table if it exists
# ------------------------------------------------------------------------------
drop_query = f"DROP TABLE IF EXISTS {args.database_name}.{args.output_table_name};"
cur.execute(drop_query)
print(f"Dropped previously existing table: {args.database_name}.{args.output_table_name} if any.")

# ------------------------------------------------------------------------------
# 2. Create Output Table
#    Note: The output_value_col_name is DOUBLE as required.
# ------------------------------------------------------------------------------
create_table_query = f"""
CREATE TABLE {args.database_name}.{args.output_table_name} (
    id BIGINT(16) UNSIGNED NOT NULL AUTO_INCREMENT,
    {args.output_group_id_col_name} VARCHAR(12),
    {args.output_feat_col_name} VARCHAR(36),
    {args.output_value_col_name} DOUBLE,
    {args.output_group_norm_col_name} DOUBLE,
    PRIMARY KEY (id),
    KEY {args.output_group_id_col_name}_idx ({args.output_group_id_col_name}),
    KEY {args.output_feat_col_name}_idx ({args.output_feat_col_name})
) DEFAULT CHARSET=utf8mb4;
"""
cur.execute(create_table_query)
print(f"Created table: {args.database_name}.{args.output_table_name}")

# ------------------------------------------------------------------------------
# 3. Read Input Table 
#    (Consider explicitly listing columns if the table order is uncertain.)
# ------------------------------------------------------------------------------
read_query = f"SELECT {args.input_group_id_col_name}, {args.input_feat_col_name}, {args.input_value_col_name}, {args.input_group_norm_col_name} FROM {args.database_name}.{args.input_table_name};"
cur.execute(read_query)
input_data = cur.fetchall()

df = pd.DataFrame(
    input_data,
    columns=[
        args.input_group_id_col_name,
        args.input_feat_col_name,
        args.input_value_col_name,
        args.input_group_norm_col_name
    ]
)
print(f"Read Input Feat Table: {args.database_name}.{args.input_table_name} of shape: {df.shape}")

# ------------------------------------------------------------------------------
# 4. Pivot the DataFrame: each row = one group; columns = feats
# ------------------------------------------------------------------------------
piv_df = df.pivot_table(
    index=args.input_group_id_col_name,
    columns=args.input_feat_col_name,
    values=args.input_value_col_name
).reset_index(drop=False)
print("Pivoted DataFrame shape:", piv_df.shape)

# ------------------------------------------------------------------------------
# 5. Dimensionality Reduction (PCA or model)
# ------------------------------------------------------------------------------
dim_red_method = args.dimensionality_reduction_method.lower()
red_piv_df = None

if dim_red_method == "pca":
    n_components = args.rfa_output_dimension or 3
    pca = PCA(n_components=n_components)

    # Separate group_id from numeric columns
    group_id_col = args.input_group_id_col_name
    numeric_cols = [col for col in piv_df.columns if col != group_id_col]

    # Fill NaNs with 0 (or apply your preferred strategy)
    X = piv_df[numeric_cols].fillna(0).values

    # Perform PCA
    X_pca = pca.fit_transform(X)  # shape: (num_rows, n_components)

    # If you prefer, you may round the values (but keep as double)
    X_pca = np.round(X_pca, decimals=8)

    # Build a new DataFrame with these PCA columns and the group_id
    pca_cols = [f"dim_feat_{i:03d}" for i in range(n_components)]
    red_piv_df = pd.DataFrame(X_pca, columns=pca_cols)
    red_piv_df[group_id_col] = piv_df[group_id_col]

    # Reorder so that group_id is the first column
    red_piv_df = red_piv_df[[group_id_col] + pca_cols]

elif dim_red_method == "model":
    # Implement your model-based dimensionality reduction technique here.
    red_piv_df = piv_df.copy()
else:
    raise ValueError(f"Unknown dimensionality_reduction_method: {args.dimensionality_reduction_method}")

# If input and output group_id column names differ, rename the column
if args.input_group_id_col_name != args.output_group_id_col_name:
    red_piv_df.rename(columns={args.input_group_id_col_name: args.output_group_id_col_name}, inplace=True)

# ------------------------------------------------------------------------------
# 6. Melt the reduced DataFrame back to long format: group_id | feat | value
#    For group_norm, here we set it equal to value; adjust if needed.
# ------------------------------------------------------------------------------
output_rfa_feat_df = red_piv_df.melt(
    id_vars=args.output_group_id_col_name,
    var_name=args.output_feat_col_name,
    value_name=args.output_value_col_name
)
output_rfa_feat_df[args.output_group_norm_col_name] = output_rfa_feat_df[args.output_value_col_name]
output_rfa_feat_df.sort_values(by=[args.output_group_id_col_name, args.output_feat_col_name], inplace=True)

print(f"Reduced DataFrame shape after melt: {output_rfa_feat_df.shape}. Preview:\n", output_rfa_feat_df.head())

# ------------------------------------------------------------------------------
# 7. Insert the data into the Output Table in batches
# ------------------------------------------------------------------------------
insert_query = f"""
INSERT INTO {args.database_name}.{args.output_table_name} (
    {args.output_group_id_col_name},
    {args.output_feat_col_name},
    {args.output_value_col_name},
    {args.output_group_norm_col_name}
) VALUES (%s, %s, %s, %s)
"""

# Prepare the data as a list of tuples.
data_to_insert = output_rfa_feat_df[[
    args.output_group_id_col_name,
    args.output_feat_col_name,
    args.output_value_col_name,
    args.output_group_norm_col_name
]].values.tolist()
print(f"Preparing to load table: {args.database_name}.{args.output_table_name} with dataframe shape: {output_rfa_feat_df.shape}")

batch_size = 1000
n_rows = len(data_to_insert)

for i in tqdm(range(0, n_rows, batch_size), desc="Inserting rows"):
    batch = data_to_insert[i:i + batch_size]
    cur.executemany(insert_query, batch)
    conn.commit()

print("Data loaded successfully.")
cur.close()
conn.close()
