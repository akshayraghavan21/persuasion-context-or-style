import argparse
import sys
import wandb
from argparse import Namespace
import json
import uuid
from datetime import datetime
import os
os.environ["WANDB_DIR"] = os.path.abspath("/data/araghavan/persuasion-context-or-style/output")
from sbertluar_ptemp_linearlayer_training_v3 import train  # Your final script's train function.

def load_default_args(cli_args):
    """
    Loads default arguments from a JSON file specified by the command-line argument
    and returns an argparse.Namespace.
    The configuration is kept in a dedicated directory (e.g., configs/).
    """
    with open(cli_args.default_config_file, "r") as f:
        config_defaults = json.load(f)
    return Namespace(**config_defaults)

# Global variable to hold the defaults.
default_args = None
args_cli = None

# Create a base sweep logs directory.
base_output_dir = os.path.join("..", "output")
os.makedirs(base_output_dir, exist_ok=True)
sweep_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Placeholders; we'll set these after obtaining the sweep_id.
sweep_id = None
sweep_folder_name = None
hyperparams_filename = None

def sweep_train():
    # Initialize the run so wandb.config is available.
    run = wandb.init(dir="../output")
    config = wandb.config

    # Merge the defaults with the sweep overrides.
    # Start with the full default config...
    full_config = dict(vars(default_args))
    # ...and update with any keys provided by the sweep.
    for key, value in config.items():
        full_config[key] = value

    print("Running training with full configuration:")
    print(json.dumps(full_config, indent=2))
    print(args_cli)

    wandb.config.update(full_config, allow_val_change=True)

    with open(hyperparams_filename, "a") as f:
        f.write(json.dumps(full_config) + "\n")

    # Create a run-specific directory inside the sweep folder.
    run_id = run.id if wandb.run is not None and hasattr(wandb.run, "id") else uuid.uuid4().hex[:6]
    run_dir_name = f"run_{sweep_timestamp}_run_{run_id}"
    run_dir = os.path.join(sweep_folder_name, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)
    full_config["run_dir"] = run_dir

    # Pass the merged configuration to your training function.
    args = Namespace(**full_config)
    train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep hyperparameter search for SBERT+LUAR linear classifier."
    )
    parser.add_argument("--config_file", type=str, required=True,
                        help="Full path to the JSON sweep configuration file.")
    parser.add_argument("--default_config_file", type=str, default="../configs/default_config.json",
                        help="Path to the default JSON configuration file.")
    args_cli = parser.parse_args()

    if not os.path.isfile(args_cli.config_file):
        print("Incorrect Config File Passed")
        sys.exit(1)

    if not os.path.isfile(args_cli.default_config_file):
        print("Default Config File not present at the specified location.")
        sys.exit(1)
    
    with open(args_cli.config_file, "r") as f:
        sweep_config = json.load(f)
    sweep_config["config_file"] = args_cli.config_file
    print(sweep_config)
    # Load and store the default args globally.
    default_args = load_default_args(args_cli)

    # Initialize the sweep in wandb using the loaded configuration.
    sweep_id = wandb.sweep(sweep_config, project="thesis_per_con_v_sty_sbert_luar_lin_clf")
    if not sweep_id:
        sweep_id = uuid.uuid4().hex[:6]

    # Create the sweep folder inside the base output directory.
    sweep_folder_name = os.path.join(base_output_dir, f"run_{sweep_timestamp}_sweep_{sweep_id}")
    os.makedirs(sweep_folder_name, exist_ok=True)
    hyperparams_filename = os.path.join(sweep_folder_name, f"sweep_hyperparams_{sweep_timestamp}_{sweep_id}.txt")

    # Launch the agent.
    wandb.agent(sweep_id, function=sweep_train)
