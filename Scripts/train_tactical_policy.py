"""
Tactical Policy Training Script for SBDAPM

Trains a PPO-based RL policy for tactical action selection using experiences
collected from Unreal Engine gameplay.

Architecture:
    Input: 71 features (FObservationElement)
    Hidden: 128 -> 128 -> 64 neurons (ReLU)
    Output: 16 actions (Softmax)

Usage:
    1. Collect experiences in Unreal and export to JSON
    2. Run: python train_tactical_policy.py --data path/to/experiences.json
    3. Model will be exported to tactical_policy.onnx
    4. Load ONNX model in Unreal via URLPolicyNetwork::LoadPolicy()

Requirements:
    pip install torch numpy tensorboard stable-baselines3
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Dict
import os
from datetime import datetime


# ============================================
# Configuration
# ============================================

class Config:
    # Network architecture
    INPUT_SIZE = 71
    HIDDEN_LAYERS = [128, 128, 64]
    OUTPUT_SIZE = 16

    # Training hyperparameters
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    GAMMA = 0.99  # Discount factor
    GAE_LAMBDA = 0.95  # GAE parameter
    CLIP_EPSILON = 0.2  # PPO clipping
    VALUE_COEF = 0.5  # Value loss coefficient
    ENTROPY_COEF = 0.01  # Entropy bonus

    # Training settings
    GRAD_CLIP = 0.5
    UPDATE_FREQUENCY = 10  # Update policy every N epochs

    # Paths
    DATA_PATH = "experiences.json"
    MODEL_OUTPUT = "tactical_policy.onnx"
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "runs"


# ============================================
# Neural Network (Policy + Value)
# ============================================

class TacticalPolicyNetwork(nn.Module):
    """
    Actor-Critic network for tactical action selection
    Outputs both action probabilities (actor) and state value (critic)
    """

    def __init__(self, input_size=71, hidden_layers=[128, 128, 64], output_size=16):
        super(TacticalPolicyNetwork, self).__init__()

        # Build shared feature extractor
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        self.feature_extractor = nn.Sequential(*layers)

        # Actor head (policy)
        self.actor = nn.Linear(prev_size, output_size)

        # Critic head (value function)
        self.critic = nn.Linear(prev_size, 1)

    def forward(self, state):
        """
        Forward pass
        Returns: (action_logits, state_value)
        """
        features = self.feature_extractor(state)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        return action_logits, state_value

    def get_action_probs(self, state):
        """Get action probabilities (for ONNX export)"""
        action_logits, _ = self.forward(state)
        return torch.softmax(action_logits, dim=-1)


# ============================================
# Experience Dataset
# ============================================

class ExperienceDataset(Dataset):
    """PyTorch dataset for experience replay"""

    def __init__(self, experiences: List[Dict]):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.terminals = []

        for exp in experiences:
            self.states.append(np.array(exp['state'], dtype=np.float32))
            self.actions.append(exp['action'])
            self.rewards.append(exp['reward'])
            self.next_states.append(np.array(exp['next_state'], dtype=np.float32))
            self.terminals.append(exp['terminal'])

        self.states = torch.FloatTensor(self.states)
        self.actions = torch.LongTensor(self.actions)
        self.rewards = torch.FloatTensor(self.rewards)
        self.next_states = torch.FloatTensor(self.next_states)
        self.terminals = torch.FloatTensor(self.terminals)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'next_state': self.next_states[idx],
            'terminal': self.terminals[idx]
        }


# ============================================
# PPO Trainer
# ============================================

class PPOTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize network
        self.policy = TacticalPolicyNetwork(
            input_size=config.INPUT_SIZE,
            hidden_layers=config.HIDDEN_LAYERS,
            output_size=config.OUTPUT_SIZE
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.LEARNING_RATE)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{config.LOG_DIR}/{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        print(f"PPOTrainer initialized on device: {self.device}")
        print(f"Network parameters: {sum(p.numel() for p in self.policy.parameters()):,}")

    def compute_gae(self, rewards, values, next_values, terminals):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.GAMMA * next_value * (1 - terminals[t]) - values[t]
            gae = delta + self.config.GAMMA * self.config.GAE_LAMBDA * (1 - terminals[t]) * gae
            advantages[t] = gae

        return advantages

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_batches = 0

        for batch in dataloader:
            states = batch['state'].to(self.device)
            actions = batch['action'].to(self.device)
            rewards = batch['reward'].to(self.device)
            next_states = batch['next_state'].to(self.device)
            terminals = batch['terminal'].to(self.device)

            # Forward pass
            action_logits, values = self.policy(states)
            _, next_values = self.policy(next_states)

            values = values.squeeze(-1)
            next_values = next_values.squeeze(-1)

            # Compute advantages
            advantages = self.compute_gae(rewards, values, next_values, terminals)
            returns = advantages + values

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy loss (PPO clip)
            action_probs = torch.softmax(action_logits, dim=-1)
            old_action_probs = action_probs.detach()

            action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)
            old_action_log_probs = torch.log(old_action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)

            ratio = torch.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.config.CLIP_EPSILON, 1.0 + self.config.CLIP_EPSILON) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(values, returns)

            # Entropy bonus (encourage exploration)
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()

            # Total loss
            loss = policy_loss + self.config.VALUE_COEF * value_loss - self.config.ENTROPY_COEF * entropy

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.GRAD_CLIP)
            self.optimizer.step()

            # Accumulate stats
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            num_batches += 1

        # Return average losses
        return {
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'entropy': total_entropy / num_batches
        }

    def train(self, dataset: ExperienceDataset):
        """Full training loop"""
        dataloader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)

        print(f"\nTraining for {self.config.NUM_EPOCHS} epochs on {len(dataset)} experiences")
        print(f"Batch size: {self.config.BATCH_SIZE}, Batches per epoch: {len(dataloader)}\n")

        for epoch in range(self.config.NUM_EPOCHS):
            self.policy.train()
            stats = self.train_epoch(dataloader)

            # Log to tensorboard
            self.writer.add_scalar('Loss/Policy', stats['policy_loss'], epoch)
            self.writer.add_scalar('Loss/Value', stats['value_loss'], epoch)
            self.writer.add_scalar('Loss/Entropy', stats['entropy'], epoch)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} | "
                      f"Policy Loss: {stats['policy_loss']:.4f} | "
                      f"Value Loss: {stats['value_loss']:.4f} | "
                      f"Entropy: {stats['entropy']:.4f}")

            # Save checkpoint
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")

        print("\nTraining complete!")

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(self.config.CHECKPOINT_DIR, filename)
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved: {path}")

    def export_onnx(self, output_path):
        """Export trained model to ONNX format"""
        self.policy.eval()

        # Create dummy input
        dummy_input = torch.randn(1, self.config.INPUT_SIZE).to(self.device)

        # Export
        torch.onnx.export(
            self.policy,
            dummy_input,
            output_path,
            input_names=['observation'],
            output_names=['action_probabilities', 'state_value'],
            dynamic_axes={'observation': {0: 'batch_size'}},
            opset_version=11
        )

        print(f"ONNX model exported to: {output_path}")


# ============================================
# Main
# ============================================

def load_experiences(json_path: str) -> List[Dict]:
    """Load experiences from JSON file"""
    print(f"Loading experiences from {json_path}...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    experiences = data['experiences']
    print(f"Loaded {len(experiences)} experiences")
    print(f"Episodes completed: {data.get('episodes_completed', 'N/A')}")
    print(f"Average reward: {data.get('average_reward', 'N/A'):.2f}")
    print(f"Best reward: {data.get('best_reward', 'N/A'):.2f}\n")

    return experiences


def main():
    parser = argparse.ArgumentParser(description='Train tactical RL policy')
    parser.add_argument('--data', type=str, default=Config.DATA_PATH, help='Path to experiences JSON')
    parser.add_argument('--output', type=str, default=Config.MODEL_OUTPUT, help='Output ONNX model path')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE, help='Batch size')

    args = parser.parse_args()

    # Update config
    config = Config()
    config.DATA_PATH = args.data
    config.MODEL_OUTPUT = args.output
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.BATCH_SIZE = args.batch_size

    # Load experiences
    experiences = load_experiences(config.DATA_PATH)
    dataset = ExperienceDataset(experiences)

    # Train
    trainer = PPOTrainer(config)
    trainer.train(dataset)

    # Export to ONNX
    trainer.export_onnx(config.MODEL_OUTPUT)

    print("\n✅ Training complete! Load the model in Unreal with:")
    print(f"   URLPolicyNetwork::LoadPolicy(\"{config.MODEL_OUTPUT}\")")


if __name__ == '__main__':
    main()
