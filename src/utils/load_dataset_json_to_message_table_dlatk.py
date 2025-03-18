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

parser.add_argument("--dataset_message_id_col", type=str, default="message_id", help="Embeddings Feat File's Corresponding Group ID Column Name")
parser.add_argument("--dataset_message_col", type=str, default="message", help="Embeddings Feat File's Corresponding Feat Column Name")
parser.add_argument("--dataset_claim_id_col", type=str, default="prompt_id", help="Embeddings Feat File's Corresponding Value Column Name")
parser.add_argument("--dataset_claim_col", type=str, default="prompt", help="Embeddings Feat File's Corresponding Group Norm Column Name")
parser.add_argument("--dataset_persuaded_col", type=str, default="persuaded", help="Embeddings Feat File's Corresponding Group Norm Column Name")
parser.add_argument("--dataset_folds_col_col", type=str, default="folds_col", help="Embeddings Feat File's Corresponding Group Norm Column Name")

parser.add_argument("--op_message_id_col_name", type=str, default="message_id", help="Output column Name: message_id")
parser.add_argument("--op_message_col_name", type=str, default="message", help="Output column Name: message")
parser.add_argument("--op_claim_id_col_name", type=str, default="claim_id", help="Output column Name: claim_id")
parser.add_argument("--op_claim_col_name", type=str, default="claim", help="Output column Name: Group claim")
parser.add_argument("--op_persuaded_col_name", type=str, default="persuaded", help="Output column Name: persuaded")
parser.add_argument("--op_folds_col_col_name", type=str, default="folds_col", help="Output column Name: folds_col")

args = parser.parse_args()
print(vars(args))

df = pd.read_json(args.input_file, lines=True)

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
    {args.op_message_id_col_name} VARCHAR(12),
    {args.op_message_col_name} TEXT,
    {args.op_claim_id_col_name} VARCHAR(12),
    {args.op_claim_col_name} TEXT,
    {args.op_persuaded_col_name} INT(1),
    {args.op_folds_col_col_name} INT(1),
    PRIMARY KEY (id),
    KEY {args.op_message_id_col_name}_idx ({args.op_message_id_col_name}),
    KEY {args.op_claim_id_col_name}_idx ({args.op_claim_id_col_name})
) DEFAULT CHARSET=utf8mb4;
"""

cur.execute(create_feat_table_query)
print(f"Created table: {args.database_name}.{args.table_name}")

## Load to Table
insert_query = f"""
INSERT INTO {args.database_name}.{args.table_name} (
    {args.op_message_id_col_name},
    {args.op_message_col_name},
    {args.op_claim_id_col_name},
    {args.op_claim_col_name},
    {args.op_persuaded_col_name},
    {args.op_folds_col_col_name}
) VALUES (%s, %s, %s, %s, %s, %s)
"""

# Prepare data as a list of tuples.
data_to_insert = df[[args.dataset_message_id_col,
                     args.dataset_message_col,
                     args.dataset_claim_id_col,
                     args.dataset_claim_col,
                     args.dataset_persuaded_col,
                     args.dataset_folds_col_col]].values.tolist()
print(f"Preparing to load to table: {args.database_name}.{args.table_name} the dataframe of shape: {df.shape}")

# Insert the data in batches
batch_size = 1000
n_rows = len(data_to_insert)

# Insert the data in batches with a tqdm progress bar
for i in tqdm(range(0, n_rows, batch_size), desc="Inserting rows"):
    batch = data_to_insert[i:i + batch_size]
    cur.executemany(insert_query, batch)
    conn.commit()

print("Data loaded successfully.")
cur.close()
conn.close()