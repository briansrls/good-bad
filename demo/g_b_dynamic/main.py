import argparse
# Ensure src modules are discoverable, e.g. by running from parent dir of demo, or adjusting PYTHONPATH
# Or, if main.py is in demo/g_b_dynamic, use relative imports if orchestrator is structured as a package point.
# For simplicity here, assuming src is in python path or we are running from a level above demo.
from src.training_orchestrator import TrainingOrchestrator, DEFAULT_CONFIG
import json # For loading config from file potentially

def main(args):
    """
    Main function to run the EOR-inspired adaptive learning experiment.
    """
    print("Starting EOR-inspired adaptive learning experiment...")

    config = DEFAULT_CONFIG
    # Load config from file if specified, overriding defaults
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                file_config = json.load(f)
            config.update(file_config) # Update defaults with file config
            print(f"Loaded configuration from {args.config_file}")
        except FileNotFoundError:
            print(f"Warning: Config file {args.config_file} not found. Using default config.")
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from {args.config_file}. Using default config.")
    
    if args.num_cycles_phase1 is not None: config['num_cycles_phase1'] = args.num_cycles_phase1
    if args.num_cycles_phase2 is not None: config['num_cycles_phase2'] = args.num_cycles_phase2
    if args.learning_rate_l1 is not None: config['learning_rate_l1'] = args.learning_rate_l1
    if args.batch_size is not None: config['batch_size'] = args.batch_size
    if args.initial_sp is not None: config['initial_sp'] = args.initial_sp

    # Initialize and run the orchestrator
    orchestrator = TrainingOrchestrator(config)
    orchestrator.run_experiment()

    print("Experiment finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EOR-Inspired Adaptive Learning Architecture")
    parser.add_argument('--config_file', type=str, default=None, help='Path to JSON configuration file.')
    parser.add_argument('--num_cycles_phase1', type=int, default=None, help='Number of cycles for Phase 1.')
    parser.add_argument('--num_cycles_phase2', type=int, default=None, help='Number of cycles for Phase 2.')
    parser.add_argument('--learning_rate_l1', type=float, default=None, help='Learning rate for L1 network.')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training.')
    parser.add_argument('--initial_sp', type=float, default=None, help='Initial survival points.')
    # Add more CLI arguments to override other config parameters as needed

    args = parser.parse_args()
    main(args) 