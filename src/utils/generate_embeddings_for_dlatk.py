from argparse import ArgumentParser
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer
import numpy as np

parser = ArgumentParser(description="generate embeddings for DLATK")

parser.add_argument("--input_file", type=str, help="Input JSONL file path")
parser.add_argument("--output_file", type=str, help="Output CSV file path")

parser.add_argument("--model_type", choices=["sbert", "luar"], help="Model type")
parser.add_argument("--model_name", type=str, help="Model name or path for generating embeddings")
parser.add_argument("--tokenizer_name", type=str, default=None, help="Tokenizer name or path for generating embeddings")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--max_length", type=int, default=512, help="Max length")
parser.add_argument("--device", type=str, default="cuda", help="Device")
parser.add_argument("--seed", type=int, default=42, help="Seed")

## Ask the user to provide the column name of which you want to generate embeddings
parser.add_argument("--text_col_name", type=str, help="Input Text Column Name to Generate Embeddings Of: eg. message, prompt etc.")
parser.add_argument("--embeddings_group_id_col", type=str, help="Input Datasets column to use for Group ID value for the embeddings: eg. message_id")
parser.add_argument("--dim_feat_prefix", type=str, default="dim_feat_", help="Prefix to the dimension feature columns")

parser.add_argument("--op_group_id_col_name", type=str, default="group_id", help="Output column Name: Group ID")
parser.add_argument("--op_feat_col_name", type=str, default="feat", help="Output column Name: Feature")
parser.add_argument("--op_value_col_name", type=str, default="value", help="Output column Name: Value")
parser.add_argument("--op_group_norm_col_name", type=str, default="group_norm", help="Output column Name: Group Norm")

args = parser.parse_args()
print(vars(args))
# ----------------------------------------------------------
# Dataset Module (token-based for both SBERT & LUAR)
# ----------------------------------------------------------
class CustomDatasetDefault(Dataset):
    def __init__(self, df: pd.DataFrame, id_col, text_col: str):
        """
        Initialize the dataset using a pandas DataFrame.
        """
        self.df = df
        self.text_col = text_col
        self.id_col = id_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row[self.id_col], row[self.text_col]


def tokenize_luar(texts, tokenizer, max_length):
    """
    Tokenize texts for LUAR-MUD.
    We still rely on the final string approach for partial prompt.
    """
    tokenized = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return tokenized


data_df = pd.read_json(args.input_file, lines=True)#[:100]
dataset = CustomDatasetDefault(data_df, args.embeddings_group_id_col, args.text_col_name)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
dataset_id_emb = []

if args.model_type == "sbert":
    model = SentenceTransformer(args.model_name)
    tokenizer = None

    for ids, texts in dataloader:
        id_np = np.array(ids).reshape(-1, 1)

        outputs = model.encode(texts, convert_to_tensor=True).to(args.device).cpu().detach().numpy()
        
        batch_id_emb = np.concatenate([id_np, outputs], axis=1)
        dataset_id_emb.append(batch_id_emb)


elif args.model_type == "luar":
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, config=config).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    for ids, texts in dataloader:
        id_np = np.array(ids).reshape(-1, 1)
        tokenized = tokenize_luar(texts, tokenizer, args.max_length)
        tokenized_luar_message = {k: v.to(args.device).unsqueeze(1) for k, v in tokenized.items()}
        
        outputs = model(**tokenized_luar_message).cpu().detach().numpy()
        
        batch_id_emb = np.concatenate([id_np, outputs], axis=1)
        dataset_id_emb.append(batch_id_emb)


dataset_id_emb = np.vstack(dataset_id_emb)
dataset_id_emb_df = pd.DataFrame(dataset_id_emb, columns=[args.op_group_id_col_name] + [f"{args.dim_feat_prefix}{i:03d}" for i in range(dataset_id_emb.shape[1]-1)])
print("Dataset ID Embeddings DF:", dataset_id_emb_df.shape)
print(dataset_id_emb_df.tail())

dataset_id_emb_melted_df = dataset_id_emb_df.melt(id_vars=[args.op_group_id_col_name], var_name=args.op_feat_col_name, value_name=args.op_value_col_name)
dataset_id_emb_melted_df.sort_values(by=[args.op_group_id_col_name, args.op_feat_col_name], inplace=True)
dataset_id_emb_melted_df[args.op_group_norm_col_name] = dataset_id_emb_melted_df[args.op_value_col_name]
print("Dataset ID Embeddings Melted DF:", dataset_id_emb_melted_df.shape)
print(dataset_id_emb_melted_df.tail())

dataset_id_emb_melted_df.to_csv(args.output_file, index=False)
print(f"Embeddings written to {args.output_file}")