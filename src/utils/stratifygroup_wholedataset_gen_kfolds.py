import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from argparse import ArgumentParser
"""
Stratification and Grouping of the dataset. Generates a test set, creates a train set with 5 folds vals
Performs the following:
    1. Reads in a dataset, stratifies it by the 'outcome_label' column, groups it by the 'prompt' column
    2. Conditional: Generates unique prompt IDs and message IDs if not present in the dataset
    3. Splits the dataset into 5 folds using StratifiedGroupKFold
    4. Held Out Test Set is chosen with the logic: In the 5 folds output of sklearns StratifiedGroupKFold, selects one fold that has the closest number of test samples to the ideal split size as the test set
    5. Chosen folds train set is further split into 5 folds using StratifiedGroupKFold
    6. A new column 'folds_col' is created in the train set to indicate which fold the row belongs to
    7. Writes the train set and test set to separate files
"""
parser = ArgumentParser(description="Stratification and Grouping of the dataset. Generates a test set, creates a train set with 5 folds vals")
parser.add_argument("--input_file", type=str, help="Input file path")
parser.add_argument("--output_train_file", type=str, help="Output train file path")
parser.add_argument("--output_held_out_test_file", type=str, help="Output test file path")
parser.add_argument("--old_label_col_name", type=str, help="Old label column name")
parser.add_argument("--new_label_col_name", type=str, help="New label column name")
parser.add_argument("--prompt_id_col_name", type=str, default="prompt_id", help="Prompt ID column name")
parser.add_argument("--message_id_col_name", type=str, default="message_id", help="Message ID column name")
parser.add_argument("--folds_col_name", type=str, default="folds_col", help="Folds column name")
args = parser.parse_args()

sgkf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)

# input_file = "/data/araghavan/persuasion-context-or-style/data/dpo_random_neg_op_comment_v003_to_per_con_v_styl_v001.json"
# output_train_file = "/data/araghavan/persuasion-context-or-style/data/dpo_random_neg_op_comment_v003_to_per_con_v_styl_v001_train_grpstrat_80pct.jsonl"
# output_held_out_test_file = "/data/araghavan/persuasion-context-or-style/data/dpo_random_neg_op_comment_v003_to_per_con_v_styl_v001_test_grpstrat_20pct.jsonl"

# old_label_col_name = "outcome_label"
# new_label_col_name = "delta_label"
# prompt_id_col_name = "prompt_id"
# message_id_col_name = "message_id"
# folds_col_name = "folds_col"

df = pd.read_json(args.input_file)
df = df.sort_values(by=['prompt'], ascending=True).reset_index(drop=True)

# Generate unique prompt IDs (fixed order for reproducibility)
unique_prompts = df['prompt'].unique()
width = len(str(len(unique_prompts)))
prompt_id_map = {prompt: f"{idx:0{width}d}" for idx, prompt in enumerate(unique_prompts)}

if args.prompt_id_col_name not in df.columns:
    df[args.prompt_id_col_name] = df['prompt'].map(prompt_id_map)
    print("Generated prompt IDs")

if args.message_id_col_name not in df.columns:
    # Generate unique message IDs (sequential, reproducible)
    width = len(str(len(df)))
    df[args.message_id_col_name] = [f"{idx:0{width}d}" for idx in range(len(df))]
    print("Generated message IDs")

if args.old_label_col_name in df.columns:
    df.rename(columns={args.old_label_col_name:args.new_label_col_name}, inplace=True)
    print(f"Renamed {args.old_label_col_name} to {args.new_label_col_name}")

folds_indices={}
ideal_fold_size = len(df) // 5
diff_len_tests_ideal_split_size_arr = []
for i, (train_index, test_index) in enumerate(sgkf.split(df, df[args.new_label_col_name], df[args.prompt_id_col_name])):
    print(f"Fold {i}")
    print("Len Train indices:", len(train_index))
    print("Len Test indices:", len(test_index))
    folds_indices[i] = (train_index, test_index, len(train_index), len(test_index))
    diff_len_tests_ideal_split_size_arr.append(abs(len(test_index)-ideal_fold_size))
    print()

best_fold_idx = np.array(diff_len_tests_ideal_split_size_arr).argmin()

train_df = df.loc[folds_indices[best_fold_idx][0]].reset_index(drop=True)

test_df = df.loc[folds_indices[best_fold_idx][1]].reset_index(drop=True)

train_val_indices = {}
for i, (train_index, val_index) in enumerate(sgkf.split(train_df, train_df[args.new_label_col_name], train_df[args.prompt_id_col_name])):
    print(f"Fold {i}")
    print("Len Train indices:", len(train_index))
    print("Len Val indices:", len(val_index))
    train_val_indices[i] = (train_index, val_index, len(train_index), len(val_index))
    print()


for fold_idx, fold_val in train_val_indices.items():
    print(f"Fold {fold_idx} {fold_val[1]}")
    train_df.loc[fold_val[1], args.folds_col_name] = fold_idx

train_df.to_json(args.output_train_file, orient='records', lines=True)
print(f"Train file written to {args.output_train_file}")

test_df.to_json(args.output_held_out_test_file, orient='records', lines=True)
print(f"Test file written to {args.output_held_out_test_file}")