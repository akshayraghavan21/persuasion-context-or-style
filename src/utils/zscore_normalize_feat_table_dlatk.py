from argparse import ArgumentParser
import pandas as pd
import MySQLdb
from tqdm import tqdm
import numpy as np
import os
from scipy.stats import zscore

# MySQL config
config_file = os.path.expanduser("~/.my.cnf")

parser = ArgumentParser(description="Z-score normalize features and reload")
parser.add_argument("--database_name", type=str, required=True)
parser.add_argument("--input_table_name", type=str, required=True)
parser.add_argument("--output_table_name", type=str, required=True)

# Column config (defaults aligned with your DB schema)
parser.add_argument("--group_id_col", type=str, default="group_id")
parser.add_argument("--feat_col", type=str, default="feat")
parser.add_argument("--value_col", type=str, default="value")
parser.add_argument("--group_norm_col", type=str, default="group_norm")

parser.add_argument("--insert_table_batch_size", type=int, default=1000)

args = parser.parse_args()
print(vars(args))

# MySQL connection
conn = MySQLdb.connect(
    db=args.database_name,
    read_default_file=config_file,
    charset="utf8mb4",
    use_unicode=True
)

# Step 1: Load the data from the input table
query = f"""
SELECT {args.group_id_col}, {args.feat_col}, {args.value_col}
FROM {args.database_name}.{args.input_table_name}
"""
df = pd.read_sql(query, conn)
print(f"Loaded {df.shape[0]} rows from {args.input_table_name}")
import pdb; pdb.set_trace()

# Step 2: Pivot the table: rows -> group_id, columns -> feat, values -> value
pivot_df = df.pivot(index=args.group_id_col, columns=args.feat_col, values=args.value_col)
print(f"Pivoted shape: {pivot_df.shape}")

# Step 3: Z-score normalization (column-wise)
zscore_df = pivot_df.apply(zscore, axis=0, nan_policy='omit').fillna(0)  # Replace NaNs if any
print("Z-score normalization complete.")

# Step 4: Melt back to long format
melted_df = zscore_df.reset_index().melt(id_vars=args.group_id_col,
                                         var_name=args.feat_col,
                                         value_name=args.value_col)
melted_df[args.group_norm_col] = melted_df[args.value_col]  # group_norm = normalized value

print(f"Melted back to shape: {melted_df.shape}")

# Step 5: Create the output table
cur = conn.cursor()
cur.execute(f"DROP TABLE IF EXISTS {args.database_name}.{args.output_table_name}")
print(f"Dropped table if existed: {args.database_name}.{args.output_table_name}")

create_table_query = f"""
CREATE TABLE {args.database_name}.{args.output_table_name} (
    id BIGINT(16) UNSIGNED NOT NULL AUTO_INCREMENT,
    {args.group_id_col} VARCHAR(20),
    {args.feat_col} VARCHAR(50),
    {args.value_col} DOUBLE,
    {args.group_norm_col} DOUBLE,
    PRIMARY KEY (id),
    KEY {args.group_id_col}_idx ({args.group_id_col}),
    KEY {args.feat_col}_idx ({args.feat_col})
) DEFAULT CHARSET=utf8mb4;
"""
cur.execute(create_table_query)
print(f"Created new table: {args.output_table_name}")

# Step 6: Prepare data for insertion
insert_query = f"""
INSERT INTO {args.database_name}.{args.output_table_name} (
    {args.group_id_col},
    {args.feat_col},
    {args.value_col},
    {args.group_norm_col}
) VALUES (%s, %s, %s, %s)
"""

data_to_insert = melted_df[[args.group_id_col, args.feat_col, args.value_col, args.group_norm_col]].values.tolist()

# Step 7: Batched insertion
batch_size = args.insert_table_batch_size
n_rows = len(data_to_insert)
for i in tqdm(range(0, n_rows, batch_size), desc="Inserting normalized rows"):
    batch = data_to_insert[i:i + batch_size]
    cur.executemany(insert_query, batch)
    conn.commit()

print("Z-score normalized data loaded successfully.")
cur.close()
conn.close()
