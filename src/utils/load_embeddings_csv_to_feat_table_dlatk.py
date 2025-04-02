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

# Expand the path to ~/.my.cnf
config_file = os.path.expanduser("~/.my.cnf")

parser = ArgumentParser(description="generate embeddings for DLATK")

parser.add_argument("--input_file", type=str, help="Input JSONL file path")
parser.add_argument("--database_name", type=str, help="Database to create feat table")
parser.add_argument("--table_name", type=str, help="Table to create to store feat values")

parser.add_argument("--embeddings_group_id_col", type=str, default="group_id", help="Embeddings Feat File's Corresponding Group ID Column Name")
parser.add_argument("--embeddings_feat_col", type=str, default="feat", help="Embeddings Feat File's Corresponding Feat Column Name")
parser.add_argument("--embeddings_value_col", type=str, default="value", help="Embeddings Feat File's Corresponding Value Column Name")
parser.add_argument("--embeddings_group_norm_col", type=str, default="group_norm", help="Embeddings Feat File's Corresponding Group Norm Column Name")

parser.add_argument("--op_group_id_col_name", type=str, default="group_id", help="Output column Name: Group ID")
parser.add_argument("--op_feat_col_name", type=str, default="feat", help="Output column Name: Feature")
parser.add_argument("--op_value_col_name", type=str, default="value", help="Output column Name: Value")
parser.add_argument("--op_group_norm_col_name", type=str, default="group_norm", help="Output column Name: Group Norm")

parser.add_argument("--insert_table_batch_size", type=int, default=1000, help="Number of Rows to insert at once in batches")

args = parser.parse_args()
print(vars(args))

df = pd.read_csv(args.input_file)

conn = MySQLdb.connect(
    db=args.database_name, 
    read_default_file=config_file, 
    charset="utf8mb4", 
    use_unicode=True
)
cur = conn.cursor()

## Drop Table If Exists
check_table_exists_query = f"DROP TABLE IF EXISTS {args.database_name}.{args.table_name};"
cur.execute(check_table_exists_query)
print(f"Dropped previously existing table: {args.database_name}.{args.table_name} if any.")

## Create Table
create_feat_table_query = f"""
CREATE TABLE {args.database_name}.{args.table_name} (
    id BIGINT(16) UNSIGNED NOT NULL AUTO_INCREMENT,
    {args.op_group_id_col_name} VARCHAR(12),
    {args.op_feat_col_name} VARCHAR(36),
    {args.op_value_col_name} DOUBLE,
    {args.op_group_norm_col_name} DOUBLE,
    PRIMARY KEY (id),
    KEY {args.op_group_id_col_name}_idx ({args.op_group_id_col_name}),
    KEY {args.op_feat_col_name}_idx ({args.op_feat_col_name})
) DEFAULT CHARSET=utf8mb4;
"""
cur.execute(create_feat_table_query)
print(f"Created table: {args.database_name}.{args.table_name}")

## Load to Table
insert_query = f"""
INSERT INTO {args.database_name}.{args.table_name} (
    {args.op_group_id_col_name},
    {args.op_feat_col_name},
    {args.op_value_col_name},
    {args.op_group_norm_col_name}
) VALUES (%s, %s, %s, %s)
"""

# Prepare data as a list of tuples.
data_to_insert = df[[args.embeddings_group_id_col,
                     args.embeddings_feat_col,
                     args.embeddings_value_col,
                     args.embeddings_group_norm_col]].values.tolist()
print(f"Preparing to load to table: {args.database_name}.{args.table_name} the dataframe of shape: {df.shape}")

# # Insert the data in batches
# cur.executemany(insert_query, data_to_insert)
# conn.commit()

# Define a batch size for insertions
batch_size = args.insert_table_batch_size if args.insert_table_batch_size else 1000
n_rows = len(data_to_insert)

# Insert the data in batches with a tqdm progress bar
for i in tqdm(range(0, n_rows, batch_size), desc="Inserting rows"):
    batch = data_to_insert[i:i + batch_size]
    cur.executemany(insert_query, batch)
    conn.commit()

print("Data loaded successfully.")
cur.close()
conn.close()
