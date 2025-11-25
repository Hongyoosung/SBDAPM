"""
World Model Training Script v3.0 for SBDAPM (Sprint 2)

Trains a neural network to predict future states for MCTS simulation.
Enables true Monte Carlo rollouts instead of static evaluation.

Architecture:
    Input: CurrentState (750) + Actions (100 encoded)
    Action Encoder: 100 -> 128 (ReLU)
    State Encoder: 750 -> 256 (ReLU)
    Fusion: 384 -> 256 -> 128 (ReLU)
    Transition Predictor: 128 -> 64 -> 750 (state delta)
    Output: NextState prediction (750 features)

Training:
    - Supervised learning on real game transitions
    - MSE loss on state prediction
    - Data collected during gameplay via FollowerAgentComponent

Usage:
    1. Enable state transition logging in Unreal
    2. Play games to collect transitions
    3. Export via FollowerAgentComponent::ExportStateTransitions()
    4. Run: python train_world_model.py --data transitions.json
    5. Model exported to world_model.onnx
    6. Load in UMCTS::InitializeTeamMCTS()

Data Format:
    {
        "transitions": [
            {
                "state_before": [...],  // 750 features
                "state_after": [...],   // 750 features
                "strategic_commands": [...],
                "tactical_actions": [...],
                "actual_delta": {...},
                "timestamp": float
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
    STATE_SIZE = 750  # TeamObservation: 40 team + 710 individual (10 agents × 71)
    ACTION_ENCODING_SIZE = 100  # Strategic commands + tactical actions encoded
    INPUT_SIZE = STATE_SIZE + ACTION_ENCODING_SIZE  # 850

    ACTION_ENCODER_SIZE = 128
    STATE_ENCODER_SIZE = 256
    FUSION_LAYERS = [384, 256, 128]
    PREDICTOR_LAYERS = [128, 64]
    OUTPUT_SIZE = STATE_SIZE  # Predict full next state

    # Training hyperparameters
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    GAMMA = 0.99  # Discount factor (not used in supervised learning)

    # Regularization
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.2

    # Training settings
    GRAD_CLIP = 1.0
    VALIDATION_SPLIT = 0.2

    # Paths
    DATA_PATH = "state_transitions.json"
    MODEL_OUTPUT = "world_model.onnx"
    CHECKPOINT_DIR = "checkpoints_world"
    LOG_DIR = "runs_world"


# ============================================
# Neural Network (World Model)
# ============================================

class WorldModel(nn.Module):
    """
    World model for state transition prediction
    Predicts next state given current state + actions
    """

    def __init__(self, config: Config):
        super(WorldModel, self).__init__()

        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(config.ACTION_ENCODING_SIZE, config.ACTION_ENCODER_SIZE),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(config.STATE_SIZE, config.STATE_ENCODER_SIZE),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

        # Fusion layers
        fusion_layers = []
        prev_size = config.ACTION_ENCODER_SIZE + config.STATE_ENCODER_SIZE
        for hidden_size in config.FUSION_LAYERS:
            fusion_layers.append(nn.Linear(prev_size, hidden_size))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(config.DROPOUT))
            prev_size = hidden_size
        self.fusion = nn.Sequential(*fusion_layers)

        # Transition predictor
        predictor_layers = []
        prev_size = config.FUSION_LAYERS[-1]
        for hidden_size in config.PREDICTOR_LAYERS:
            predictor_layers.append(nn.Linear(prev_size, hidden_size))
            predictor_layers.append(nn.ReLU())
            prev_size = hidden_size
        predictor_layers.append(nn.Linear(prev_size, config.OUTPUT_SIZE))
        self.predictor = nn.Sequential(*predictor_layers)

    def forward(self, state, actions):
        """
        Forward pass
        Args:
            state: Current state tensor [batch_size, state_size]
            actions: Actions tensor [batch_size, action_encoding_size]
        Returns:
            next_state: Predicted next state [batch_size, state_size]
        """
        # Encode inputs
        state_encoded = self.state_encoder(state)
        action_encoded = self.action_encoder(actions)

        # Fuse encodings
        fused = torch.cat([state_encoded, action_encoded], dim=1)
        fused = self.fusion(fused)

        # Predict next state
        next_state = self.predictor(fused)

        return next_state


# ============================================
# Dataset
# ============================================

class StateTransitionDataset(Dataset):
    """Dataset of state transitions for world model training"""

    def __init__(self, data_path: str, config: Config):
        self.config = config

        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)

        self.transitions = data.get('transitions', [])

        print(f"Loaded {len(self.transitions)} state transitions from {data_path}")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        transition = self.transitions[idx]

        # State before
        state_before = np.array(transition['state_before'], dtype=np.float32)

        # Pad/truncate to STATE_SIZE
        if len(state_before) < self.config.STATE_SIZE:
            state_before = np.pad(state_before, (0, self.config.STATE_SIZE - len(state_before)))
        elif len(state_before) > self.config.STATE_SIZE:
            state_before = state_before[:self.config.STATE_SIZE]

        # State after (target)
        state_after = np.array(transition['state_after'], dtype=np.float32)

        # Pad/truncate to STATE_SIZE
        if len(state_after) < self.config.STATE_SIZE:
            state_after = np.pad(state_after, (0, self.config.STATE_SIZE - len(state_after)))
        elif len(state_after) > self.config.STATE_SIZE:
            state_after = state_after[:self.config.STATE_SIZE]

        # Actions (placeholder encoding - in full implementation, encode commands + tactical actions)
        actions = np.random.randn(self.config.ACTION_ENCODING_SIZE).astype(np.float32)  # Placeholder

        return (
            torch.from_numpy(state_before),
            torch.from_numpy(actions),
            torch.from_numpy(state_after)
        )


# ============================================
# Training Loop
# ============================================

def train_world_model(config: Config, args):
    """Train the world model"""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Load dataset
    dataset = StateTransitionDataset(args.data, config)

    # Split into train/validation
    val_size = int(len(dataset) * config.VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"Train samples: {train_size}, Validation samples: {val_size}")

    # Create model
    model = WorldModel(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.LOG_DIR, f"world_model_{timestamp}")
    writer = SummaryWriter(log_dir)

    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0

        for batch_idx, (state_before, actions, state_after) in enumerate(train_loader):
            state_before = state_before.to(device)
            actions = actions.to(device)
            state_after = state_after.to(device)

            # Forward pass
            predicted_state = model(state_before, actions)
            loss = criterion(predicted_state, state_after)

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
            for state_before, actions, state_after in val_loader:
                state_before = state_before.to(device)
                actions = actions.to(device)
                state_after = state_after.to(device)

                predicted_state = model(state_before, actions)
                loss = criterion(predicted_state, state_after)
                mae = torch.abs(predicted_state - state_after).mean()

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
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_world_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

    # Load best model
    checkpoint = torch.load(os.path.join(config.CHECKPOINT_DIR, "best_world_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nBest model from epoch {checkpoint['epoch']+1} (val_loss: {checkpoint['val_loss']:.4f})")

    # Export to ONNX
    export_to_onnx(model, config, args.output)

    writer.close()
    print(f"\nTraining complete! Model saved to {args.output}")


def export_to_onnx(model: nn.Module, config: Config, output_path: str):
    """Export trained model to ONNX format for Unreal Engine NNE"""

    model.eval()

    # Create dummy inputs
    dummy_state = torch.randn(1, config.STATE_SIZE)
    dummy_actions = torch.randn(1, config.ACTION_ENCODING_SIZE)

    # Export
    torch.onnx.export(
        model,
        (dummy_state, dummy_actions),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['current_state', 'actions'],
        output_names=['predicted_next_state'],
        dynamic_axes={
            'current_state': {0: 'batch_size'},
            'actions': {0: 'batch_size'},
            'predicted_next_state': {0: 'batch_size'}
        }
    )

    print(f"Model exported to ONNX: {output_path}")
    print(f"  Input 1: current_state [{config.STATE_SIZE}]")
    print(f"  Input 2: actions [{config.ACTION_ENCODING_SIZE}]")
    print(f"  Output: predicted_next_state [{config.OUTPUT_SIZE}]")


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Train World Model for State Prediction')
    parser.add_argument('--data', type=str, default=Config.DATA_PATH,
                        help='Path to state transition data JSON')
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
    print("World Model Training (Sprint 2)")
    print("=" * 60)
    print(f"State size: {config.STATE_SIZE}")
    print(f"Action encoding: {config.ACTION_ENCODING_SIZE}")
    print(f"Architecture: State={config.STATE_ENCODER_SIZE}, Action={config.ACTION_ENCODER_SIZE}")
    print(f"              Fusion={config.FUSION_LAYERS}, Predictor={config.PREDICTOR_LAYERS}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print("=" * 60)

    # Train
    train_world_model(config, args)


if __name__ == '__main__':
    main()
