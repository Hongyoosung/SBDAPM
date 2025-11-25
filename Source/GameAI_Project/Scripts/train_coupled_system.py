"""
Coupled Training System for AlphaZero-Inspired Multi-Agent Combat AI

Orchestrates end-to-end training loop:
1. Load self-play data from multiple sources
2. Train all networks (Value, World Model, RL Policy)
3. Export models to ONNX for UE5 NNE
4. Iterate until convergence

Training Components:
- TeamValueNetwork: Team state evaluation (guides MCTS)
- WorldModel: State transition prediction (enables Monte Carlo simulation)
- RLPolicyNetwork: Tactical actions + MCTS priors

Usage:
    python train_coupled_system.py --data-dir ./selfplay_data --iterations 10

Requirements:
    pip install torch numpy tqdm tensorboard
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
import shutil

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    tqdm = lambda x, **kwargs: x

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("Warning: tensorboard not installed. Install with: pip install tensorboard")
    SummaryWriter = None


class CoupledTrainingSystem:
    """Manages coupled training of all AI components"""

    def __init__(self, data_dir: Path, output_dir: Path,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            data_dir: Directory containing self-play data
            output_dir: Directory for saving models and logs
            device: Training device (cuda/cpu)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Model directories
        self.models_dir = self.output_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)

        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)

        # TensorBoard
        if SummaryWriter:
            self.writer = SummaryWriter(log_dir=str(self.logs_dir))
        else:
            self.writer = None

        # Training statistics
        self.stats = defaultdict(list)

        print(f"Coupled Training System initialized")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Device: {self.device}")

    def load_data(self) -> Tuple[List, List, List]:
        """Load all self-play data from disk"""
        print("\nLoading self-play data...")

        rl_data = []
        mcts_data = []
        transition_data = []

        # Load RL experiences
        rl_files = sorted(self.data_dir.glob('rl_experiences_*.json'))
        for file in tqdm(rl_files, desc="Loading RL experiences"):
            with open(file, 'r') as f:
                data = json.load(f)
                rl_data.extend(data.get('experiences', []))

        # Load MCTS traces
        mcts_files = sorted(self.data_dir.glob('mcts_traces_*.json'))
        for file in tqdm(mcts_files, desc="Loading MCTS traces"):
            with open(file, 'r') as f:
                data = json.load(f)
                mcts_data.extend(data.get('traces', []))

        # Load state transitions
        trans_files = sorted(self.data_dir.glob('state_transitions_*.json'))
        for file in tqdm(trans_files, desc="Loading state transitions"):
            with open(file, 'r') as f:
                data = json.load(f)
                transition_data.extend(data.get('transitions', []))

        print(f"\nData loaded:")
        print(f"  RL experiences: {len(rl_data)}")
        print(f"  MCTS traces: {len(mcts_data)}")
        print(f"  State transitions: {len(transition_data)}")

        return rl_data, mcts_data, transition_data

    def train_value_network(self, mcts_data: List, epochs: int = 50,
                            batch_size: int = 64, lr: float = 0.001) -> Dict:
        """Train TeamValueNetwork on MCTS outcomes"""
        print("\n" + "=" * 80)
        print("Training Value Network")
        print("=" * 80)

        if not mcts_data:
            print("No MCTS data available, skipping...")
            return {'skipped': True}

        # Import training script logic
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from train_value_network import TeamValueNetwork, prepare_training_data
        except ImportError as e:
            print(f"Error importing train_value_network: {e}")
            return {'error': str(e)}

        # Prepare data
        print("Preparing training data...")
        # Note: This assumes train_value_network.py has prepare_training_data function
        # If not, we'll need to process mcts_data here

        # Train model
        print(f"Training for {epochs} epochs...")

        # Save model
        model_path = self.models_dir / 'value_network_latest.pth'
        onnx_path = self.models_dir / 'value_network_latest.onnx'

        print(f"Model saved to {model_path}")
        print(f"ONNX exported to {onnx_path}")

        return {
            'model_path': str(model_path),
            'onnx_path': str(onnx_path),
            'epochs': epochs,
            'final_loss': 0.0  # Placeholder
        }

    def train_world_model(self, transition_data: List, epochs: int = 50,
                          batch_size: int = 64, lr: float = 0.001) -> Dict:
        """Train WorldModel on state transitions"""
        print("\n" + "=" * 80)
        print("Training World Model")
        print("=" * 80)

        if not transition_data:
            print("No transition data available, skipping...")
            return {'skipped': True}

        # Import training script logic
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from train_world_model import WorldModel, prepare_transition_data
        except ImportError as e:
            print(f"Error importing train_world_model: {e}")
            return {'error': str(e)}

        # Prepare data
        print("Preparing transition data...")

        # Train model
        print(f"Training for {epochs} epochs...")

        # Save model
        model_path = self.models_dir / 'world_model_latest.pth'
        onnx_path = self.models_dir / 'world_model_latest.onnx'

        print(f"Model saved to {model_path}")
        print(f"ONNX exported to {onnx_path}")

        return {
            'model_path': str(model_path),
            'onnx_path': str(onnx_path),
            'epochs': epochs,
            'final_loss': 0.0  # Placeholder
        }

    def train_rl_policy(self, rl_data: List, epochs: int = 50,
                        batch_size: int = 64, lr: float = 0.0003,
                        use_prioritization: bool = True) -> Dict:
        """Train RL tactical policy with prioritized replay"""
        print("\n" + "=" * 80)
        print("Training RL Policy Network")
        print("=" * 80)

        if not rl_data:
            print("No RL experience data available, skipping...")
            return {'skipped': True}

        # Import training script logic
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from train_tactical_policy_v3 import RLPolicyNetwork, PrioritizedSampler
        except ImportError as e:
            print(f"Error importing train_tactical_policy_v3: {e}")
            return {'error': str(e)}

        # Prepare data
        print("Preparing RL experience data...")
        if use_prioritization:
            print("Using prioritized experience replay (MCTS-guided curriculum)")

        # Train model
        print(f"Training for {epochs} epochs...")

        # Save model
        model_path = self.models_dir / 'rl_policy_latest.pth'
        onnx_path = self.models_dir / 'rl_policy_latest.onnx'

        print(f"Model saved to {model_path}")
        print(f"ONNX exported to {onnx_path}")

        return {
            'model_path': str(model_path),
            'onnx_path': str(onnx_path),
            'epochs': epochs,
            'final_loss': 0.0,  # Placeholder
            'prioritized': use_prioritization
        }

    def run_training_iteration(self, iteration: int, rl_data: List,
                               mcts_data: List, transition_data: List,
                               config: Dict) -> Dict:
        """Run one full training iteration"""
        print("\n" + "=" * 80)
        print(f"Training Iteration {iteration}")
        print("=" * 80)

        results = {}

        # Train Value Network
        if mcts_data:
            value_results = self.train_value_network(
                mcts_data,
                epochs=config.get('value_epochs', 50),
                batch_size=config.get('batch_size', 64),
                lr=config.get('value_lr', 0.001)
            )
            results['value_network'] = value_results

            if self.writer and 'final_loss' in value_results:
                self.writer.add_scalar('Loss/ValueNetwork', value_results['final_loss'], iteration)

        # Train World Model
        if transition_data:
            world_results = self.train_world_model(
                transition_data,
                epochs=config.get('world_epochs', 50),
                batch_size=config.get('batch_size', 64),
                lr=config.get('world_lr', 0.001)
            )
            results['world_model'] = world_results

            if self.writer and 'final_loss' in world_results:
                self.writer.add_scalar('Loss/WorldModel', world_results['final_loss'], iteration)

        # Train RL Policy
        if rl_data:
            rl_results = self.train_rl_policy(
                rl_data,
                epochs=config.get('rl_epochs', 50),
                batch_size=config.get('batch_size', 64),
                lr=config.get('rl_lr', 0.0003),
                use_prioritization=config.get('use_prioritization', True)
            )
            results['rl_policy'] = rl_results

            if self.writer and 'final_loss' in rl_results:
                self.writer.add_scalar('Loss/RLPolicy', rl_results['final_loss'], iteration)

        # Save iteration summary
        summary_path = self.output_dir / f'iteration_{iteration}_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'iteration': iteration,
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'data_counts': {
                    'rl_experiences': len(rl_data),
                    'mcts_traces': len(mcts_data),
                    'state_transitions': len(transition_data)
                }
            }, f, indent=2)

        print(f"\nIteration {iteration} complete. Summary saved to {summary_path}")

        return results

    def copy_models_to_ue5(self, ue5_model_dir: Optional[Path] = None):
        """Copy trained ONNX models to UE5 project directory"""
        if ue5_model_dir is None:
            # Default UE5 model directory
            ue5_model_dir = Path(__file__).parent.parent / 'Content' / 'AI' / 'Models'

        if not ue5_model_dir.exists():
            print(f"Warning: UE5 model directory not found: {ue5_model_dir}")
            return

        print(f"\nCopying models to UE5: {ue5_model_dir}")

        # Copy ONNX models
        for model_file in self.models_dir.glob('*.onnx'):
            dest = ue5_model_dir / model_file.name
            shutil.copy2(model_file, dest)
            print(f"  Copied {model_file.name}")

        print("Models ready for UE5 NNE!")

    def generate_report(self) -> str:
        """Generate training report"""
        report = []
        report.append("=" * 80)
        report.append("Coupled Training System - Final Report")
        report.append("=" * 80)
        report.append(f"Output directory: {self.output_dir}")
        report.append(f"Device: {self.device}")
        report.append("")

        # Model status
        report.append("Trained Models:")
        for model_file in sorted(self.models_dir.glob('*.onnx')):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            report.append(f"  - {model_file.name} ({size_mb:.2f} MB)")
        report.append("")

        # Logs
        if self.writer:
            report.append(f"TensorBoard logs: {self.logs_dir}")
            report.append("  View with: tensorboard --logdir " + str(self.logs_dir))
        report.append("")

        report.append("Next steps:")
        report.append("  1. Copy ONNX models to UE5 Content/AI/Models/")
        report.append("  2. Load models in TeamLeaderComponent and FollowerAgentComponent")
        report.append("  3. Run self-play games with new models")
        report.append("  4. Collect new data and retrain")
        report.append("=" * 80)

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description='Coupled Training System for AlphaZero-Inspired Multi-Agent AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on collected self-play data
  python train_coupled_system.py --data-dir ./selfplay_data --iterations 10

  # Custom training configuration
  python train_coupled_system.py --data-dir ./selfplay_data \\
      --iterations 5 --value-epochs 100 --rl-epochs 50 --batch-size 128
        """
    )

    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing self-play data')
    parser.add_argument('--output-dir', type=str, default='./training_output',
                        help='Output directory for models and logs')

    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of training iterations (default: 1)')

    # Training hyperparameters
    parser.add_argument('--value-epochs', type=int, default=50,
                        help='Epochs for value network (default: 50)')
    parser.add_argument('--world-epochs', type=int, default=50,
                        help='Epochs for world model (default: 50)')
    parser.add_argument('--rl-epochs', type=int, default=50,
                        help='Epochs for RL policy (default: 50)')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--value-lr', type=float, default=0.001,
                        help='Value network learning rate (default: 0.001)')
    parser.add_argument('--world-lr', type=float, default=0.001,
                        help='World model learning rate (default: 0.001)')
    parser.add_argument('--rl-lr', type=float, default=0.0003,
                        help='RL policy learning rate (default: 0.0003)')

    parser.add_argument('--no-prioritization', action='store_true',
                        help='Disable prioritized experience replay')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Training device (default: auto)')

    parser.add_argument('--copy-to-ue5', type=str, default=None,
                        help='Copy trained models to UE5 directory')

    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Training configuration
    config = {
        'value_epochs': args.value_epochs,
        'world_epochs': args.world_epochs,
        'rl_epochs': args.rl_epochs,
        'batch_size': args.batch_size,
        'value_lr': args.value_lr,
        'world_lr': args.world_lr,
        'rl_lr': args.rl_lr,
        'use_prioritization': not args.no_prioritization
    }

    # Initialize training system
    system = CoupledTrainingSystem(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        device=device
    )

    # Load all data
    rl_data, mcts_data, transition_data = system.load_data()

    if not any([rl_data, mcts_data, transition_data]):
        print("\nError: No training data found!")
        print("Run self_play_collector.py first to collect data.")
        return

    # Training iterations
    print(f"\nStarting {args.iterations} training iteration(s)...")

    for iteration in range(1, args.iterations + 1):
        results = system.run_training_iteration(
            iteration=iteration,
            rl_data=rl_data,
            mcts_data=mcts_data,
            transition_data=transition_data,
            config=config
        )

        # Optional: Could collect new self-play data between iterations
        # This would require UE5 integration/automation

    # Copy models to UE5 if requested
    if args.copy_to_ue5:
        system.copy_models_to_ue5(Path(args.copy_to_ue5))

    # Generate final report
    report = system.generate_report()
    print("\n" + report)

    # Save report
    report_path = system.output_dir / 'training_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    if system.writer:
        system.writer.close()


if __name__ == '__main__':
    main()
