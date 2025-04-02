from argparse import ArgumentParser
import pandas as pd
import MySQLdb
import sys
import os
from tqdm import tqdm

# Expand the path to ~/.my.cnf
config_file = os.path.expanduser("~/.my.cnf")

parser = ArgumentParser(description="Generate embeddings table for DLATK with optional UTCs")

parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file path")
parser.add_argument("--database_name", type=str, required=True, help="Database to create feat table")
parser.add_argument("--table_name", type=str, required=True, help="Table to create to store feat values")
parser.add_argument("--format_type", type=str, choices=["cmv_v001", "cmv_v002"], required=True,
                    help="Dataset format: 'cmv_v001' (no UTCs) or 'cmv_v002' (includes UTCs)")

# Dataset Columns (default for CMV)
parser.add_argument("--dataset_message_id_col", type=str, default="message_id")
parser.add_argument("--dataset_message_col", type=str, default="message")
parser.add_argument("--dataset_claim_id_col", type=str, default="prompt_id")
parser.add_argument("--dataset_claim_col", type=str, default="prompt")
parser.add_argument("--dataset_persuaded_col", type=str, default="persuaded")
parser.add_argument("--dataset_folds_col_col", type=str, default="folds_col")

# Created UTC columns (optional in cmv_v002)
parser.add_argument("--dataset_message_created_col", type=str, default="message_created_utc")
parser.add_argument("--dataset_claim_created_col", type=str, default="claim_created_utc")

# Output Column Names
parser.add_argument("--op_message_id_col_name", type=str, default="message_id")
parser.add_argument("--op_message_col_name", type=str, default="message")
parser.add_argument("--op_claim_id_col_name", type=str, default="claim_id")
parser.add_argument("--op_claim_col_name", type=str, default="claim")
parser.add_argument("--op_persuaded_col_name", type=str, default="persuaded")
parser.add_argument("--op_folds_col_col_name", type=str, default="folds_col")
parser.add_argument("--op_message_created_col_name", type=str, default="message_created_utc")
parser.add_argument("--op_claim_created_col_name", type=str, default="claim_created_utc")

args = parser.parse_args()
print(vars(args))

# Load data
df = pd.read_json(args.input_file, lines=True)

# Check if UTCs are expected based on format type
include_utcs = args.format_type == "cmv_v002"

# If UTCs are expected but columns are missing, initialize as None
if include_utcs:
    for col in [args.dataset_message_created_col, args.dataset_claim_created_col]:
        if col not in df.columns:
            print(f"Warning: Column {col} missing in data. Filling with None.")
            df[col] = None
else:
    # If cmv_v001, add dummy UTC columns as None for schema consistency
    df[args.dataset_message_created_col] = None
    df[args.dataset_claim_created_col] = None

# MySQL Connection
conn = MySQLdb.connect(
    db=args.database_name,
    read_default_file=config_file,
    charset="utf8mb4",
    use_unicode=True
)
cur = conn.cursor()

# Drop existing table
cur.execute(f"DROP TABLE IF EXISTS {args.database_name}.{args.table_name};")
print(f"Dropped table if existed: {args.database_name}.{args.table_name}")

# Create table (UTC fields included but allow NULL)
create_feat_table_query = f"""
CREATE TABLE {args.database_name}.{args.table_name} (
    id BIGINT(16) UNSIGNED NOT NULL AUTO_INCREMENT,
    {args.op_message_id_col_name} VARCHAR(20),
    {args.op_message_col_name} TEXT,
    {args.op_claim_id_col_name} VARCHAR(20),
    {args.op_claim_col_name} TEXT,
    {args.op_persuaded_col_name} INT(1),
    {args.op_folds_col_col_name} INT(1),
    {args.op_message_created_col_name} BIGINT NULL,
    {args.op_claim_created_col_name} BIGINT NULL,
    PRIMARY KEY (id),
    KEY {args.op_message_id_col_name}_idx ({args.op_message_id_col_name}),
    KEY {args.op_claim_id_col_name}_idx ({args.op_claim_id_col_name})
) DEFAULT CHARSET=utf8mb4;
"""
cur.execute(create_feat_table_query)
print(f"Created table: {args.database_name}.{args.table_name}")

# Insert query (UTCs always present, but NULL if missing)
insert_query = f"""
INSERT INTO {args.database_name}.{args.table_name} (
    {args.op_message_id_col_name},
    {args.op_message_col_name},
    {args.op_claim_id_col_name},
    {args.op_claim_col_name},
    {args.op_persuaded_col_name},
    {args.op_folds_col_col_name},
    {args.op_message_created_col_name},
    {args.op_claim_created_col_name}
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

# Prepare data rows
data_to_insert = df[[args.dataset_message_id_col,
                     args.dataset_message_col,
                     args.dataset_claim_id_col,
                     args.dataset_claim_col,
                     args.dataset_persuaded_col,
                     args.dataset_folds_col_col,
                     args.dataset_message_created_col,
                     args.dataset_claim_created_col]].values.tolist()

print(f"Loading {len(data_to_insert)} rows into table: {args.database_name}.{args.table_name}")

# Insert in batches
batch_size = 1000
for i in tqdm(range(0, len(data_to_insert), batch_size), desc="Inserting rows"):
    batch = data_to_insert[i:i + batch_size]
    cur.executemany(insert_query, batch)
    conn.commit()

print("Data loaded successfully.")
cur.close()
conn.close()
