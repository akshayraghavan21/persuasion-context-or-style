import argparse
import json
import os
import random
import sys
import uuid
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sklearn.metrics import accuracy_score, roc_auc_score

import wandb  # pip install wandb
from datasets import load_dataset, DatasetDict  # pip install datasets

# -----------------------------
# Data Splitting Module
# -----------------------------
def load_and_prepare_dataset_no_leaks(args: argparse.Namespace) -> DatasetDict:
    """
    Load and prepare the dataset, ensuring no data leakage between splits.
    This function loads the dataset from one JSON file, groups examples by prompt,
    and then splits the groups into train, validation, and test sets.
    """
    dataset_file = os.path.abspath(os.path.join(args.project_data_dir, args.data_file))
    if not os.path.isfile(dataset_file):
        print("Dataset file does not exist. Aborting.")
        sys.exit(1)
    
    print(f"Loading Dataset File: {dataset_file}")
    dataset = load_dataset("json", data_files=dataset_file, split="train")

    # Group indices by prompt.
    prompt_groups = {record['prompt']: [] for record in dataset}
    for idx, record in enumerate(dataset):
        prompt_groups[record['prompt']].append(idx)
    
    # Use fixed seed for reproducibility.
    random.seed(args.seed)
    prompt_keys = list(prompt_groups.keys())
    random.shuffle(prompt_keys)
    
    total_groups = len(prompt_keys)
    train_size = int(0.6 * total_groups)
    val_size = int(0.2 * total_groups)
    
    train_groups = prompt_keys[:train_size]
    val_groups = prompt_keys[train_size:train_size+val_size]
    test_groups = prompt_keys[train_size+val_size:]
    
    def get_indices(groups):
        return [idx for group in groups for idx in prompt_groups[group]]
    
    return DatasetDict({
        'train': dataset.select(get_indices(train_groups)),
        'validation': dataset.select(get_indices(val_groups)),
        'test': dataset.select(get_indices(test_groups)),
    })

# -----------------------------
# Dataset Module
# -----------------------------
class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, include_prompt=False, max_length=512):
        """
        Initialize the dataset using a pandas DataFrame.
        """
        self.df = df
        self.include_prompt = include_prompt
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        message = row["message"]
        if self.include_prompt:
            prompt = row.get("prompt", "")
            text = prompt + " " + message
        else:
            text = message
        label = int(row["outcome_label"])
        return text, label

# -----------------------------
# Tokenization Helper
# -----------------------------
def tokenize_luar(texts, tokenizer, max_length):
    """
    Tokenize texts for LUAR-MUD.
    """
    tokenized = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return tokenized

# -----------------------------
# Model Module
# -----------------------------
class SBERTLUARClassifier(nn.Module):
    def __init__(self, sbert_model, luar_model, classifier_output_dim=2, combine_method='average', only_luar=False):
        """
        only_luar: If True, use only LUAR-MUD's embedding (ignoring SBERT).
        classifier_output_dim: The output dimension of the final linear layer.
        """
        super(SBERTLUARClassifier, self).__init__()
        self.only_luar = only_luar
        self.combine_method = combine_method
        
        self.luar = luar_model

        if not self.only_luar:
            self.sbert = sbert_model
            self.sbert_dim = self.sbert.get_sentence_embedding_dimension()
        else:
            self.sbert = None
            self.sbert_dim = None

        # Fetch LUAR hidden size from its configuration.
        self.luar_dim = luar_model.config.embedding_size

        # When combining, project SBERT embedding into LUAR space if needed.
        if not self.only_luar:
            if self.sbert_dim != self.luar_dim:
                self.proj = nn.Linear(self.sbert_dim, self.luar_dim)
            else:
                self.proj = None
            classifier_input_dim = self.luar_dim
        else:
            classifier_input_dim = self.luar_dim
            self.proj = None

        self.classifier = nn.Linear(classifier_input_dim, classifier_output_dim)

    def forward(self, texts, tokenizer_luar, device, pretokenized_luar=None, max_length=512):
        if not self.only_luar:
            with torch.no_grad():
                sbert_emb = self.sbert.encode(texts, convert_to_tensor=True).to(device)
            if self.proj is not None:
                sbert_emb = self.proj(sbert_emb)
        else:
            sbert_emb = None

        # Use pretokenized LUAR inputs if provided; otherwise, tokenize here.
        if pretokenized_luar is None:
            tokenized = tokenize_luar(texts, tokenizer_luar, max_length)
            tokenized = {k: v.to(device).unsqueeze(1) for k, v in tokenized.items()}
        else:
            tokenized = pretokenized_luar

        # Forward pass through LUAR-MUD.
        luar_emb = self.luar(**tokenized)
        # # Mean-pool the last hidden state over the sequence dimension.
        # luar_emb = luar_outputs.last_hidden_state.mean(dim=1)

        if not self.only_luar:
            if self.combine_method == 'average':
                combined = torch.mean(torch.stack([sbert_emb, luar_emb]), dim=0)
            elif self.combine_method == 'sum':
                combined = sbert_emb + luar_emb
            else:
                raise ValueError(f"Unknown combine_method: {self.combine_method}")
        else:
            combined = luar_emb

        logits = self.classifier(combined)
        return logits

# -----------------------------
# Evaluation & Training Modules
# -----------------------------
def evaluate(model, data_loader, tokenizer_luar, device, max_length):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for texts, labels in data_loader:
            labels_tensor = torch.as_tensor(labels, dtype=torch.long).to(device)
            # Pre-tokenize LUAR inputs outside the forward pass.
            tokenized_luar = tokenize_luar(texts, tokenizer_luar, max_length)
            tokenized_luar = {k: v.to(device).unsqueeze(1) for k, v in tokenized_luar.items()}
            logits = model(texts, tokenizer_luar, device, pretokenized_luar=tokenized_luar, max_length=max_length)
            loss = criterion(logits, labels_tensor)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(labels)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        roc_auc = 0.0
    return avg_loss, accuracy, roc_auc

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # If only_luar flag is set, mark that no SBERT model is used.
    if args.only_luar:
        args.sbert_model_name = "None"
    
    # Load and split the dataset using a fixed seed.
    dataset_dict = load_and_prepare_dataset_no_leaks(args)
    # Convert splits to pandas DataFrames.
    train_df = dataset_dict["train"].to_pandas()
    val_df = dataset_dict["validation"].to_pandas()
    test_df = dataset_dict["test"].to_pandas()

    # Initialize wandb.
    wandb.init(project=args.wandb_project, config=vars(args))

    # Load LUAR model.
    config = AutoConfig.from_pretrained(args.luar_model_name, trust_remote_code=True)
    luar_model = AutoModel.from_pretrained(args.luar_model_name, config=config, trust_remote_code=True)
    for param in luar_model.parameters():
        param.requires_grad = False
    luar_model.to(device)
    luar_model.eval()

    tokenizer_luar = AutoTokenizer.from_pretrained(args.luar_model_name, trust_remote_code=True)

    # Load SBERT model only if needed.
    if not args.only_luar:
        sbert_model = SentenceTransformer(args.sbert_model_name)
        sbert_model.eval()
        for param in sbert_model.parameters():
            param.requires_grad = False
    else:
        sbert_model = None

    # Create model.
    model = SBERTLUARClassifier(
        sbert_model, 
        luar_model, 
        classifier_output_dim=args.classifier_output_dim,
        combine_method=args.combine_method,
        only_luar=args.only_luar
    )
    model.to(device)

    # Only train the classifier.
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Create DataLoaders using our DataFrames.
    train_dataset = CustomDataset(train_df, include_prompt=args.include_prompt, max_length=args.max_length)
    eval_dataset = CustomDataset(val_df, include_prompt=args.include_prompt, max_length=args.max_length)
    test_dataset = CustomDataset(test_df, include_prompt=args.include_prompt, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # For local metrics saving.
    run_metrics = {"epochs": []}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        for texts, labels in train_loader:
            labels_tensor = torch.as_tensor(labels, dtype=torch.long).to(device)
            optimizer.zero_grad()
            # Pre-tokenize LUAR inputs outside the forward pass.
            tokenized_luar = tokenize_luar(texts, tokenizer_luar, max_length=args.max_length)
            tokenized_luar = {k: v.to(device).unsqueeze(1) for k, v in tokenized_luar.items()}
            logits = model(texts, tokenizer_luar, device, pretokenized_luar=tokenized_luar, max_length=args.max_length)
            loss = criterion(logits, labels_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_train_labels.extend(labels)
            all_train_preds.extend(preds.cpu().numpy())

        train_loss = epoch_loss / len(train_loader)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        
        eval_loss, eval_acc, _ = evaluate(model, eval_loader, tokenizer_luar, device, max_length=args.max_length)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
        }
        run_metrics["epochs"].append(epoch_metrics)
        wandb.log(epoch_metrics)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.4f}")

    # Final test evaluation.
    test_loss, test_acc, test_roc_auc = evaluate(model, test_loader, tokenizer_luar, device, max_length=args.max_length)
    final_metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_roc_auc": test_roc_auc,
    }
    run_metrics["final"] = final_metrics
    wandb.log(final_metrics)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | Test ROC AUC: {test_roc_auc:.4f}")

    # Save run metrics locally with a unique filename.
    os.makedirs(args.metrics_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    metrics_filename = os.path.join(args.metrics_dir, f"run_metrics_{timestamp}_{unique_id}.json")
    with open(metrics_filename, "w") as f:
        json.dump(run_metrics, f, indent=4)
    print(f"Run metrics saved to {metrics_filename}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SBERT+LUAR classifier with wandb logging and local metrics saving."
    )
    parser.add_argument("--project_data_dir", type=str, default="../data/", help="Directory containing the dataset file.")
    parser.add_argument("--data_file", type=str, default="dpo_random_neg_op_comment_v003_to_per_con_v_styl_v001.json", help="Name of the JSON file containing the dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization.")
    parser.add_argument("--combine_method", type=str, default="average", choices=["average", "sum"], help="Method to combine embeddings.")
    parser.add_argument("--include_prompt", action="store_true", help="If set, include prompt text concatenated to message as model input.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 penalty) for optimizer (ridge regression).")
    parser.add_argument("--wandb_project", type=str, default="thesis_per_con_v_sty_sbert_luar_lin_clf", help="wandb project name.")
    parser.add_argument("--metrics_dir", type=str, default="../output/metrics", help="Directory to save intermediate metrics files.")
    parser.add_argument("--only_luar", action="store_true", help="If set, use only LUAR-MUD embeddings (ignoring SBERT).")
    parser.add_argument("--classifier_output_dim", type=int, default=2, help="Output dimension of the classifier (default 2).")
    parser.add_argument("--sbert_model_name", type=str, default="all-distilroberta-v1", help="Name or path of the SBERT model.")
    parser.add_argument("--luar_model_name", type=str, default="rrivera1849/LUAR-MUD", help="Name or path of the LUAR model.")
    args = parser.parse_args()
    train(args)
