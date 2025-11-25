"""
Team Value Network Training Script v3.0 for SBDAPM

Trains a value network to estimate team state values for MCTS leaf node evaluation.
Replaces hand-crafted heuristics with learned value function.

Architecture:
    Input: Team observation (40 team + N×71 individual features)
    Embedding: 256 neurons (ReLU)
    Trunk: 256 -> 256 -> 128 (ReLU)
    Value Head: 128 -> 64 -> 1 (Tanh)
    Output: State value [-1, 1] (loss → win probability)

Training:
    - Supervised learning on MCTS rollout outcomes
    - TD-learning with bootstrapping
    - MSE loss on value prediction

Usage:
    1. Collect MCTS rollout data in Unreal and export to JSON
    2. Run: python train_value_network.py --data path/to/mcts_rollouts.json
    3. Model will be exported to team_value_network.onnx
    4. Load ONNX model in UMCTS::InitializeTeamMCTS()

Data Format:
    {
        "rollouts": [
            {
                "team_observation": [...],  // Flattened team obs
                "outcome": 1.0,             // Game outcome: 1.0 (win), -1.0 (loss), 0.0 (draw)
                "visit_counts": [...],      // MCTS visit counts
                "value_estimate": 0.5       // MCTS final value estimate
            },
            ...
        ]
    }

Requirements:
    pip install torch numpy tensorboard onnx
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict
import os
from datetime import datetime


# ============================================
# Configuration
# ============================================

class Config:
    # Network architecture
    MAX_AGENTS = 10
    INDIVIDUAL_FEATURES = 71
    TEAM_FEATURES = 40
    INPUT_SIZE = TEAM_FEATURES + (MAX_AGENTS * INDIVIDUAL_FEATURES)  # 40 + 710 = 750

    EMBEDDING_SIZE = 256
    TRUNK_LAYERS = [256, 256, 128]
    VALUE_HEAD = [128, 64, 1]

    # Training hyperparameters
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    GAMMA = 0.99  # Discount factor for TD-learning

    # Regularization
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.3

    # Training settings
    GRAD_CLIP = 1.0
    VALIDATION_SPLIT = 0.2

    # Paths
    DATA_PATH = "mcts_rollouts.json"
    MODEL_OUTPUT = "team_value_network.onnx"
    CHECKPOINT_DIR = "checkpoints_value"
    LOG_DIR = "runs_value"


# ============================================
# Neural Network (Team Value Network)
# ============================================

class TeamValueNetwork(nn.Module):
    """
    Value network for team state evaluation
    Maps team observation to scalar value in [-1, 1]
    """

    def __init__(self, config: Config):
        super(TeamValueNetwork, self).__init__()

        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(config.INPUT_SIZE, config.EMBEDDING_SIZE),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

        # Shared trunk
        trunk_layers = []
        prev_size = config.EMBEDDING_SIZE
        for hidden_size in config.TRUNK_LAYERS:
            trunk_layers.append(nn.Linear(prev_size, hidden_size))
            trunk_layers.append(nn.ReLU())
            trunk_layers.append(nn.Dropout(config.DROPOUT))
            prev_size = hidden_size
        self.trunk = nn.Sequential(*trunk_layers)

        # Value head
        value_layers = []
        prev_size = config.TRUNK_LAYERS[-1]
        for hidden_size in config.VALUE_HEAD[:-1]:
            value_layers.append(nn.Linear(prev_size, hidden_size))
            value_layers.append(nn.ReLU())
            prev_size = hidden_size
        value_layers.append(nn.Linear(prev_size, config.VALUE_HEAD[-1]))
        value_layers.append(nn.Tanh())  # Output in [-1, 1]
        self.value_head = nn.Sequential(*value_layers)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Team observation tensor [batch_size, input_size]
        Returns:
            value: Scalar value in [-1, 1] [batch_size, 1]
        """
        x = self.embedding(x)
        x = self.trunk(x)
        value = self.value_head(x)
        return value


# ============================================
# Dataset
# ============================================

class MCTSRolloutDataset(Dataset):
    """Dataset of MCTS rollouts for value network training"""

    def __init__(self, data_path: str, config: Config):
        self.config = config

        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)

        self.rollouts = data.get('rollouts', [])

        print(f"Loaded {len(self.rollouts)} rollouts from {data_path}")

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        rollout = self.rollouts[idx]

        # Team observation (flattened)
        team_obs = np.array(rollout['team_observation'], dtype=np.float32)

        # Pad to INPUT_SIZE if needed
        if len(team_obs) < self.config.INPUT_SIZE:
            team_obs = np.pad(team_obs, (0, self.config.INPUT_SIZE - len(team_obs)))
        elif len(team_obs) > self.config.INPUT_SIZE:
            team_obs = team_obs[:self.config.INPUT_SIZE]

        # Target value (game outcome or MCTS estimate)
        outcome = rollout.get('outcome', rollout.get('value_estimate', 0.0))
        target_value = np.array([outcome], dtype=np.float32)

        return torch.from_numpy(team_obs), torch.from_numpy(target_value)


# ============================================
# Training Loop
# ============================================

def train_value_network(config: Config, args):
    """Train the team value network"""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Load dataset
    dataset = MCTSRolloutDataset(args.data, config)

    # Split into train/validation
    val_size = int(len(dataset) * config.VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"Train samples: {train_size}, Validation samples: {val_size}")

    # Create model
    model = TeamValueNetwork(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.LOG_DIR, f"value_network_{timestamp}")
    writer = SummaryWriter(log_dir)

    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0

        for batch_idx, (team_obs, target_value) in enumerate(train_loader):
            team_obs = team_obs.to(device)
            target_value = target_value.to(device)

            # Forward pass
            predicted_value = model(team_obs)
            loss = criterion(predicted_value, target_value)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        val_mae = 0.0

        with torch.no_grad():
            for team_obs, target_value in val_loader:
                team_obs = team_obs.to(device)
                target_value = target_value.to(device)

                predicted_value = model(team_obs)
                loss = criterion(predicted_value, target_value)
                mae = torch.abs(predicted_value - target_value).mean()

                val_loss += loss.item()
                val_mae += mae.item()

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Logging
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_value_network.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

    # Load best model
    checkpoint = torch.load(os.path.join(config.CHECKPOINT_DIR, "best_value_network.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nBest model from epoch {checkpoint['epoch']+1} (val_loss: {checkpoint['val_loss']:.4f})")

    # Export to ONNX
    export_to_onnx(model, config, args.output)

    writer.close()
    print(f"\nTraining complete! Model saved to {args.output}")


def export_to_onnx(model: nn.Module, config: Config, output_path: str):
    """Export trained model to ONNX format for Unreal Engine NNE"""

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, config.INPUT_SIZE)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['team_observation'],
        output_names=['state_value'],
        dynamic_axes={
            'team_observation': {0: 'batch_size'},
            'state_value': {0: 'batch_size'}
        }
    )

    print(f"Model exported to ONNX: {output_path}")
    print(f"  Input: team_observation [{config.INPUT_SIZE}]")
    print(f"  Output: state_value [1]")


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Train Team Value Network')
    parser.add_argument('--data', type=str, default=Config.DATA_PATH,
                        help='Path to MCTS rollout data JSON')
    parser.add_argument('--output', type=str, default=Config.MODEL_OUTPUT,
                        help='Output ONNX model path')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                        help='Learning rate')

    args = parser.parse_args()

    # Update config
    config = Config()
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr

    print("=" * 60)
    print("Team Value Network Training")
    print("=" * 60)
    print(f"Input size: {config.INPUT_SIZE}")
    print(f"Architecture: {config.INPUT_SIZE} -> {config.EMBEDDING_SIZE} -> {config.TRUNK_LAYERS} -> {config.VALUE_HEAD}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print("=" * 60)

    # Train
    train_value_network(config, args)


if __name__ == '__main__':
    main()
