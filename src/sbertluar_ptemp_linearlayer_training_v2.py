import argparse
import json
import os
import random
import sys
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sklearn.metrics import accuracy_score, roc_auc_score
import copy

import wandb
from datasets import load_dataset, DatasetDict

# ----------------------------------------------------------
# Data Splitting Module (same as before)
# ----------------------------------------------------------
def load_and_prepare_dataset_no_leaks(args: argparse.Namespace) -> dict:
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
    
    # Use fixed seed for reproducibility in splitting.
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
    
    return {
        'train': dataset.select(get_indices(train_groups)),
        'validation': dataset.select(get_indices(val_groups)),
        'test': dataset.select(get_indices(test_groups)),
    }


# ----------------------------------------------------------
# Dataset Module (token-based for both SBERT & LUAR)
# ----------------------------------------------------------
class CustomDatasetDefault(Dataset):
    def __init__(self, df: pd.DataFrame, max_length=512):
        """
        Initialize the dataset using a pandas DataFrame.
        """
        self.df = df
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        message = row["message"]
        prompt = row.get("prompt", "")
        label = int(row["outcome_label"])
        return prompt, message, label

# ----------------------------------------------------------
# Tokenization Helper for LUAR (unchanged)
# ----------------------------------------------------------
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


# ----------------------------------------------------------
# Model Module: SBERT+LUAR Classifier
# ----------------------------------------------------------
class SBERTLUARClassifier(nn.Module):
    def __init__(
            self,
            sbert_model, 
            luar_model, 
            classifier_output_dim=2, 
            combine_method='average', 
            only_luar=False, 
            include_prompt=False):
        """
        only_luar: If True, use only LUAR-MUD's embedding (ignoring SBERT).
        include_prompt: If True, then combine message and prompt embeddings.
        combine_method: How to combine SBERT and LUAR embeddings.
        classifier_output_dim: Output dimension of the final linear layer (default 2 for binary).
        """
        super(SBERTLUARClassifier, self).__init__()
        self.luar = luar_model
        self.combine_method = combine_method
        self.only_luar = only_luar
        self.include_prompt = include_prompt

        if not self.only_luar:
            self.sbert = sbert_model
            self.sbert_dim = self.sbert.get_sentence_embedding_dimension()
        else:
            self.sbert = None
            self.sbert_dim = None

        # LUAR hidden size from config
        self.luar_dim = luar_model.config.embedding_size

        # Possibly project SBERT embedding into LUAR dimension
        if not self.only_luar:
            if self.sbert_dim != self.luar_dim:
                print("Projecting SBERT embeddings to match LUAR dimension.")
                self.proj = nn.Linear(self.sbert_dim, self.luar_dim)
            else:
                self.proj = None
            classifier_input_dim = self.luar_dim
        else:
            classifier_input_dim = self.luar_dim
            self.proj = None
        
        if self.include_prompt:
            classifier_input_dim *= 2

        self.classifier = nn.Linear(classifier_input_dim, classifier_output_dim)

    def forward(self, 
                sbert_message,         # list of raw strings for SBERT
                sbert_prompt,
                luar_tokenized_message,          # list of raw strings for LUAR
                luar_tokenized_prompt, 
                device,
                max_length=512):        
        # 1) SBERT embeddings (if not only_luar)
        if not self.only_luar:
            with torch.no_grad():
                sbert_emb_message = self.sbert.encode(sbert_message, convert_to_tensor=True).to(device)
                if self.include_prompt:
                    sbert_emb_prompt = self.sbert.encode(sbert_prompt, convert_to_tensor=True).to(device)
                
            if self.proj is not None:
                sbert_emb_message = self.proj(sbert_emb_message)
                if self.include_prompt:
                    sbert_emb_prompt= self.proj(sbert_emb_prompt)
            
            sbert_emb = sbert_emb_message
            if self.include_prompt:
                sbert_emb = torch.cat([sbert_emb_prompt, sbert_emb_message], dim=1)
                
        else:
            sbert_emb = None

        # 2) LUAR embeddings
        luar_emb_message = self.luar(**luar_tokenized_message)
        luar_emb = luar_emb_message
        if self.include_prompt:
            luar_emb_prompt = self.luar(**luar_tokenized_prompt)
            luar_emb = torch.cat([luar_emb_prompt, luar_emb_message], dim=1)

        # 3) Combine
        if not self.only_luar:
            if self.combine_method == 'average':
                combined = torch.mean(torch.stack([sbert_emb, luar_emb]), dim=0)
            elif self.combine_method == 'sum':
                combined = sbert_emb + luar_emb
            else:
                raise ValueError(f"Unknown combine_method: {self.combine_method}")
        else:
            combined = luar_emb

        # 4) Classify
        logits = self.classifier(combined)
        return logits


# ----------------------------------------------------------
# Evaluation
# ----------------------------------------------------------
def evaluate(model, data_loader, tokenizer_luar, device, max_length):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for prompt, message, labels in data_loader:
            labels_tensor = torch.as_tensor(labels, dtype=torch.long).to(device)

            # Pre-tokenize LUAR inputs outside the forward pass
            tokenized_luar_message = tokenize_luar(message, tokenizer_luar, max_length=max_length)
            tokenized_luar_message = {k: v.to(device).unsqueeze(1) for k, v in tokenized_luar_message.items()}

            tokenized_luar_prompt = tokenize_luar(prompt, tokenizer_luar, max_length=max_length)
            tokenized_luar_prompt = {k: v.to(device).unsqueeze(1) for k, v in tokenized_luar_prompt.items()}

            logits = model(
                message,
                prompt,
                tokenized_luar_message,
                tokenized_luar_prompt,
                device,
                max_length=max_length
            )
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
    except ValueError:
        roc_auc = 0.0
    return avg_loss, accuracy, roc_auc


# ----------------------------------------------------------
# Training Module with Early Stopping + min_delta
# ----------------------------------------------------------
def train(args):
    """
    Train function that handles model preparation, data loading, 
    training, validation, test evaluation, and logging with wandb.
    Implements optional early stopping based on validation loss (if patience > 0).
    The improvement must exceed --min_delta to reset the early-stopping counter.

    Now using token-based partial prompt for both SBERT and LUAR.
    """

    # -----------------------------
    # Reproducibility
    # -----------------------------
    import torch.backends.cudnn as cudnn
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If only_luar flag is set, mark that no SBERT model is used.
    if args.only_luar:
        args.sbert_model_name = "None"

    # Load and split the dataset using a fixed seed.
    dataset_dict = load_and_prepare_dataset_no_leaks(args)
    # Convert splits to pandas
    train_df = dataset_dict["train"].to_pandas()
    val_df = dataset_dict["validation"].to_pandas()
    test_df = dataset_dict["test"].to_pandas()

    # Initialize wandb
    wandb.init(project=args.wandb_project, config=vars(args), dir="../output")

    # Determine or create the run_dir.
    # If args.run_dir is already provided (e.g. via the hyperparameter sweep), use it.
    # Otherwise, create a new run_dir under args.output_dir.
    if not hasattr(args, "run_dir") or not args.run_dir:
        # Try to use wandb run id if available; otherwise, generate a new uuid.
        run_id = None
        if wandb.run is not None and hasattr(wandb.run, "id"):
            run_id = wandb.run.id
        else:
            run_id = uuid.uuid4().hex[:6]
        print(f"Using run ID: {run_id}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir_name = f"run_{timestamp}_run_{run_id}"
        # Use args.output_dir as parent (default provided via command-line argument)
        default_base = args.output_dir  # Note: in the argument parser, use --output_dir (not metrics_dir)
        run_dir = os.path.join(default_base, run_dir_name)
        os.makedirs(run_dir, exist_ok=True)
        args.run_dir = run_dir
    else:
        run_dir = args.run_dir

    # Print hyperparameters for clarity.
    print("Hyperparameters for this run:")
    print(json.dumps(vars(args), indent=2))
    print(f"Dataset Shapes:\n\tTrain: {train_df.shape}\n\tEval: {val_df.shape}\n\tTest: {test_df.shape}")

    # -----------------------------
    # Load LUAR
    # -----------------------------
    luar_config = AutoConfig.from_pretrained(args.luar_model_name, trust_remote_code=True)
    luar_model = AutoModel.from_pretrained(args.luar_model_name, config=luar_config, trust_remote_code=True)
    for param in luar_model.parameters():
        param.requires_grad = False
    luar_model.to(device)
    luar_model.eval()

    tokenizer_luar = AutoTokenizer.from_pretrained(args.luar_model_name, trust_remote_code=True)

    # -----------------------------
    # Load SBERT (if not only_luar)
    # -----------------------------
    if not args.only_luar:
        sbert_model = SentenceTransformer(args.sbert_model_name)
        sbert_model.eval()
        for param in sbert_model.parameters():
            param.requires_grad = False
    else:
        sbert_model = None

    # --------------------------------------------------------
    # Check if tokenizer_luar.model_max_length matches args.max_length
    # --------------------------------------------------------
    if not args.only_luar and tokenizer_luar.model_max_length != sbert_model.max_seq_length:
        raise ValueError(
            f"SBERT max_seq_length={sbert_model.max_seq_length}, "
            f"but LUAR Tokenizer's model_max_length is {tokenizer_luar.model_max_length}. They must match!"
        )
    
    # -----------------------------
    # Build SBERT+LUAR classifier
    # -----------------------------
    model = SBERTLUARClassifier(
        sbert_model, 
        luar_model, 
        classifier_output_dim=args.classifier_output_dim,
        combine_method=args.combine_method,
        only_luar=args.only_luar
    )
    model.to(device)

    # Only train the classifier head
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # Build Token-Based Dataset
    # -----------------------------
    train_dataset = CustomDatasetDefault(
        train_df,
        max_length=args.max_length,
    )
    eval_dataset = CustomDatasetDefault(
        val_df,
        max_length=args.max_length,
    )
    test_dataset = CustomDatasetDefault(
        test_df,
        max_length=args.max_length,
    )

    # We get (sbert_text, luar_text, label) from each dataset item
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader  = DataLoader(eval_dataset, batch_size=args.batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size)

    run_metrics = {"config": vars(args), "epochs": []}
    # -----------------------------
    # Early Stopping Setup
    # -----------------------------
    use_early_stopping = (args.patience > 0)
    best_eval_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    no_improvement_count = 0
    stopped_early = False
    best_model_path = None
    last_model_path = None

    # -----------------------------
    # Main Training Loop
    # -----------------------------
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        print_text = True

        for param in model.luar.parameters():
            if param.requires_grad:
                print(f"Epoch {epoch+1}: LUAR is being trained.")
                break

        for prompt, message, labels in train_loader:
            # if print_text:
            #    print_text = False
            #    print(f"Epoch:{epoch}\n\n\t\tsbert_texts:{sbert_texts[0]}\n\n\t\tluar_texts:{luar_texts[0]}\n\n\t\tlabels:{labels[0]}")

            labels_tensor = torch.as_tensor(labels, dtype=torch.long).to(device)

            optimizer.zero_grad()
            # Pre-tokenize LUAR
            tokenized_luar_message = tokenize_luar(message, tokenizer_luar, max_length=args.max_length)
            tokenized_luar_message = {k: v.to(device).unsqueeze(1) for k, v in tokenized_luar_message.items()}

            tokenized_luar_prompt = tokenize_luar(prompt, tokenizer_luar, max_length=args.max_length)
            tokenized_luar_prompt = {k: v.to(device).unsqueeze(1) for k, v in tokenized_luar_prompt.items()}

            logits = model(
                message,
                prompt,
                tokenized_luar_message,
                tokenized_luar_prompt,
                device,
                max_length=args.max_length
            )
            loss = criterion(logits, labels_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_train_labels.extend(labels)
            all_train_preds.extend(preds.cpu().numpy())

        train_loss = epoch_loss / len(train_loader)
        train_acc = accuracy_score(all_train_labels, all_train_preds)

        # Validation step
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

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.4f}")

        # Check for improvement in validation loss
        if eval_loss + args.min_delta < best_eval_loss:
            # Save best model so far
            best_eval_loss = eval_loss
            no_improvement_count = 0
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            best_model_path = os.path.join(run_dir, f"best_model_epoch_{best_epoch}.pt")
            torch.save(best_model_state, best_model_path)
        else:
            no_improvement_count += 1

        # Early Stopping Check
        if use_early_stopping and no_improvement_count >= args.patience:
            print(f"Early stopping triggered at epoch {epoch+1} "
                  f"(no improvement in {args.patience} epochs, min_delta={args.min_delta}).")
            wandb.log({"early_stopping_epoch": epoch + 1})
            stopped_early = True
            break

    print("-" * 50)
    if stopped_early:
        print(f"Stopped early after epoch {epoch+1}. Best model from epoch {best_epoch} was saved to {best_model_path}.")
    else:
        print(f"Finished all {args.epochs} epochs. Best model from epoch {best_epoch} was saved to {best_model_path}.")
        last_model_path = os.path.join(run_dir, f"last_model_epoch_{args.epochs}.pt")
        torch.save(model.state_dict(), last_model_path)
        print(f"Last epoch model saved to {last_model_path}.")

    # -----------------------------
    # Load the Best Model, Evaluate on Test
    # -----------------------------
    if best_model_state is not None and args.load_best_model and best_model_path is not None:
        print(f"Loading best model weights from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path))

    eval_loss, eval_acc, _ = evaluate(model, eval_loader, tokenizer_luar, device, max_length=args.max_length)
    final_eval_metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "eval_roc_auc": _,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_model_path": best_model_path,
        "last_model_path": last_model_path
    }
    run_metrics["final_eval_metrics"] = final_eval_metrics

    # -----------------------------
    # Final Test Evaluation
    # -----------------------------
    test_loss, test_acc, test_roc_auc = evaluate(model, test_loader, tokenizer_luar, device, max_length=args.max_length)
    final_metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_roc_auc": test_roc_auc,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_model_path": best_model_path,
        "last_model_path": last_model_path
    }
    run_metrics["final"] = final_metrics
    wandb.log(final_metrics)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | ROC AUC: {test_roc_auc:.4f}")
    if stopped_early:
        print("Note: The model stopped early (see wandb logs).")

    # Save run metrics locally
    metrics_filename = os.path.join(run_dir, "run_metrics.json")
    with open(metrics_filename, "w") as f:
        json.dump(run_metrics, f, indent=4)
    print(f"Run metrics saved to {metrics_filename}")

    wandb.finish()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SBERT+LUAR classifier with wandb logging and local metrics saving, using token-based partial prompt for both."
    )
    # First add argument for the default config file.
    parser.add_argument("--default_config", type=str, default="../configs/default_config.json",
                        help="Path to the default JSON config file (e.g., ../configs/default_config.json).")
    # Parse known arguments to load the default configuration.
    temp_args, remaining_argv = parser.parse_known_args()
    if not os.path.isfile(temp_args.default_config):
        print(f"Default config file not found at {temp_args.default_config}. Aborting.")
        sys.exit(1)
    with open(temp_args.default_config, "r") as f:
        defaults = json.load(f)
    parser.set_defaults(**defaults)
    
    # Add remaining command-line arguments (overrides the defaults if provided).
    parser.add_argument("--project_data_dir", type=str, help="Directory containing the dataset file.")
    parser.add_argument("--data_file", type=str, help="Name of the JSON file containing the dataset.")
    parser.add_argument("--seed", type=int, help="Random seed for splitting and training.")
    parser.add_argument("--lr", type=float, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, help="Max number of training epochs.")
    parser.add_argument("--max_length", type=int, help="Maximum sequence length for tokenization.")
    parser.add_argument("--combine_method", type=str, choices=["average", "sum"], help="Method to combine embeddings.")
    parser.add_argument("--include_prompt", action="store_true", help="If set, do partial prompt + full message at token level for both SBERT & LUAR.")
    parser.add_argument("--weight_decay", type=float, help="Weight decay (L2 penalty).")
    parser.add_argument("--wandb_project", type=str, help="wandb project name.")
    parser.add_argument("--output_dir", type=str, help="Directory to save run data.")
    parser.add_argument("--only_luar", action="store_true", help="If set, use only LUAR embeddings (ignore SBERT).")
    parser.add_argument("--classifier_output_dim", type=int, help="Classifier output dimension (default=2).")
    parser.add_argument("--sbert_model_name", type=str, help="Name/path of the SBERT model.")
    parser.add_argument("--luar_model_name", type=str, help="Name/path of the LUAR model.")
    parser.add_argument("--patience", type=int, help="Epochs to wait for improvement before early stopping. If <= 0, disabled.")
    parser.add_argument("--min_delta", type=float, help="Min absolute improvement in val_loss to reset early-stopping counter.")
    parser.add_argument("--no_load_best_model", action="store_false", dest="load_best_model",
                        help="If set, do NOT load the best model at the end of training (default: load).")

    args = parser.parse_args()
    train(args)
