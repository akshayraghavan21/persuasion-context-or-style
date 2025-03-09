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

parser.add_argument("--op_group_id_col_name", type=str, default="message_id", help="Output column Name: Group ID")
parser.add_argument("--op_feat_col_name", type=str, default="feat", help="Output column Name: Feature")
parser.add_argument("--op_value_col_name", type=str, default="value", help="Output column Name: Value")
parser.add_argument("--op_group_norm_col_name", type=str, default="group_norm", help="Output column Name: Group Norm")

args = parser.parse_args()
print(vars(args))

df = pd.read_json(args.input_file, lines=True)

conn = MySQLdb.connect(
    db=args.database, 
    read_default_file=config_file, 
    charset="utf8mb4", 
    use_unicode=True
)
cur = conn.cursor()

## Drop Table If Exists
check_table_exists_query = f"DROP TABLE IF EXISTS {args.database_name}.{args.table_name};"
cur.execute(check_table_exists_query)

## Create Table
create_feat_table_query = f"""
CREATE TABLE {args.database_name}.{args.table_name} (
    id BIGINT(16) UNSIGNED NOT NULL AUTO_INCREMENT,
    {args.op_message_id_col_name} INT(11),
    {args.op_message_col_name} TEXT,
    {args.op_claim_id_col_name} INT(11),
    {args.op_claim_col_name} TEXT,
    {args.op_persuaded_col_name} INT(1),
    PRIMARY KEY (id),
    KEY {args.op_message_id_col_name}_idx ({args.op_message_id_col_name}),
    KEY {args.op_claim_id_col_name}_idx ({args.op_claim_id_col_name})
) DEFAULT CHARSET=utf8;
"""

cur.execute(create_feat_table_query)

## Load to Table
insert_query = f"""
INSERT INTO {args.database_name}.{args.table_name} (
    {args.op_message_id_col_name},
    {args.op_message_col_name},
    {args.op_claim_id_col_name},
    {args.op_claim_col_name},
    {args.op_persuaded_col_name}
) VALUES (%s, %s, %s, %s, %s)
"""

# Prepare data as a list of tuples.
data_to_insert = df[[args.op_message_id_col_name,
                     args.op_message_col_name,
                     args.op_claim_id_col_name,
                     args.op_claim_col_name,
                     args.op_persuaded_col_name]].values.tolist()

# Insert the data in batches
cur.executemany(insert_query, data_to_insert)
conn.commit()

print("Data loaded successfully.")
cur.close()
conn.close()