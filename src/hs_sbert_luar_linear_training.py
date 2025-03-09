import argparse
import sys
import wandb
from argparse import Namespace
import json
import uuid
from datetime import datetime
import os

# Import your training function from your module.
from sbert_luar_linear_training import train

def create_default_args():
    """
    Creates a default argparse.Namespace with the same defaults as in your training code.
    """
    args = Namespace()
    args.project_data_dir = "../data/"
    args.data_file = "dpo_random_neg_op_comment_v003_to_per_con_v_styl_v001.json"
    args.seed = 42
    args.lr = 1e-3
    args.batch_size = 32  # default (will be updated by sweep)
    args.epochs = 10      # default (will be updated by sweep)
    args.max_length = 512
    args.combine_method = "average"  # fixed as average
    args.include_prompt = False
    args.weight_decay = 0.0  # default (will be updated by sweep)
    args.wandb_project = "thesis_per_con_v_sty_sbert_luar_lin_clf"
    args.metrics_dir = "metrics"
    args.only_luar = False  # fixed as False
    args.classifier_output_dim = 2
    args.sbert_model_name = "all-distilroberta-v1"  # default (will be updated by sweep)
    args.luar_model_name = "rrivera1849/LUAR-MUD"
    return args

# The sweep configuration will be loaded from a config file.
# Create a unique file name for logging hyperparameter configurations in ../output/sweep_logs/
output_dir = os.path.join("..", "output", "sweep_logs")
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
unique_id = uuid.uuid4().hex[:6]
hyperparams_filename = os.path.join(output_dir, f"sweep_hyperparams_{timestamp}_{unique_id}.txt")

def sweep_train():
    # Initialize the run. This ensures wandb.config is available.
    run = wandb.init()
    config = wandb.config

    # Get the default args.
    args = create_default_args()
    # Update args with the current sweep configuration.
    args.lr = config.lr
    args.batch_size = config.batch_size
    args.epochs = config.epochs
    args.combine_method = config.combine_method  # always "average" if that is fixed
    args.weight_decay = config.weight_decay
    args.only_luar = config.only_luar          # always False if fixed
    args.sbert_model_name = config.sbert_model_name

    # Optionally, print the updated hyperparameters.
    print("Running training with hyperparameters:")
    print(f"lr: {args.lr}, batch_size: {args.batch_size}, epochs: {args.epochs}, "
          f"combine_method: {args.combine_method}, weight_decay: {args.weight_decay}, "
          f"only_luar: {args.only_luar}, sbert_model_name: {args.sbert_model_name}")

    # Write the current hyperparameter configuration along with the run ID to the unique file.
    run_info = {
        "run_id": run.id,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "combine_method": args.combine_method,
        "weight_decay": args.weight_decay,
        "only_luar": args.only_luar,
        "sbert_model_name": args.sbert_model_name
    }
    with open(hyperparams_filename, "a") as f:
        f.write(json.dumps(run_info) + "\n")

    # Call the train function from your training code.
    train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep hyperparameter search for SBERT+LUAR linear classifier using a config file."
    )
    parser.add_argument("--config_file", type=str, required=True,
                        help="Full path to the JSON sweep configuration file")
    args_cli = parser.parse_args()

    # Load the sweep configuration from the provided file.
    if not os.path.isfile(args_cli.config_file):
        print("Incorrect Config File Passed")
        sys.exit(1)
        
    with open(args_cli.config_file, "r") as f:
        sweep_config = json.load(f)

    # Initialize the sweep in wandb using the loaded configuration.
    sweep_id = wandb.sweep(sweep_config, project="thesis_per_con_v_sty_sbert_luar_lin_clf")
    
    # Launch the agent.
    wandb.agent(sweep_id, function=sweep_train)
