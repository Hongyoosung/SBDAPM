"""
RLlib Training Script for SBDAPM

Trains PPO agents via Schola gRPC connection to Unreal Engine.

Usage:
    1. Start UE with Schola plugin (game mode)
    2. Run: python train_rllib.py
    3. Model exports to tactical_policy.onnx

Requirements:
    pip install schola[rllib] ray[rllib] torch
"""

import argparse
import os
import sys
from datetime import datetime

# Check for required packages
try:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import Policy
    RLLIB_AVAILABLE = True
except ImportError:
    RLLIB_AVAILABLE = False
    print("Error: ray[rllib] not installed. Run: pip install ray[rllib]")

try:
    from schola.envs import UnrealEnv
    from schola.scripts.utils import make_env_creator
    SCHOLA_AVAILABLE = True
except ImportError:
    SCHOLA_AVAILABLE = False
    print("Warning: schola not installed. Run: pip install schola[rllib]")

import numpy as np
from gymnasium import spaces


class SBDAPMConfig:
    """Training configuration."""

    # Environment
    HOST = "localhost"
    PORT = 50051
    MAX_EPISODE_STEPS = 1000

    # Network architecture (matches train_tactical_policy.py)
    HIDDEN_LAYERS = [128, 128, 64]

    # PPO hyperparameters
    LEARNING_RATE = 3e-4
    TRAIN_BATCH_SIZE = 4000
    SGD_MINIBATCH_SIZE = 128
    NUM_SGD_ITER = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_PARAM = 0.2
    ENTROPY_COEFF = 0.01
    VF_LOSS_COEFF = 0.5

    # Training
    NUM_WORKERS = 0  # Use main process for UE connection
    NUM_ENVS_PER_WORKER = 1
    NUM_ITERATIONS = 100
    CHECKPOINT_FREQ = 10

    # Paths
    OUTPUT_DIR = "training_results"
    MODEL_NAME = "tactical_policy"


def create_env_config():
    """Create environment configuration for Schola."""
    return {
        "host": SBDAPMConfig.HOST,
        "port": SBDAPMConfig.PORT,
        "max_episode_steps": SBDAPMConfig.MAX_EPISODE_STEPS,
    }


def create_ppo_config():
    """Create RLlib PPO configuration."""
    config = (
        PPOConfig()
        .environment(
            env="sbdapm_env",
            env_config=create_env_config(),
        )
        .framework("torch")
        .training(
            lr=SBDAPMConfig.LEARNING_RATE,
            train_batch_size=SBDAPMConfig.TRAIN_BATCH_SIZE,
            sgd_minibatch_size=SBDAPMConfig.SGD_MINIBATCH_SIZE,
            num_sgd_iter=SBDAPMConfig.NUM_SGD_ITER,
            gamma=SBDAPMConfig.GAMMA,
            lambda_=SBDAPMConfig.GAE_LAMBDA,
            clip_param=SBDAPMConfig.CLIP_PARAM,
            entropy_coeff=SBDAPMConfig.ENTROPY_COEFF,
            vf_loss_coeff=SBDAPMConfig.VF_LOSS_COEFF,
            model={
                "fcnet_hiddens": SBDAPMConfig.HIDDEN_LAYERS,
                "fcnet_activation": "relu",
            },
        )
        .env_runners(
            num_env_runners=SBDAPMConfig.NUM_WORKERS,
            num_envs_per_env_runner=SBDAPMConfig.NUM_ENVS_PER_WORKER,
        )
        .debugging(log_level="INFO")
    )
    return config


def register_env():
    """Register custom environment with Ray."""
    from ray.tune.registry import register_env

    if SCHOLA_AVAILABLE:
        # Use Schola's UnrealEnv with our configuration
        def env_creator(config):
            from sbdapm_env import SBDAPMScholaEnv
            return SBDAPMScholaEnv(**config)
    else:
        # Fallback to dummy env for testing
        def env_creator(config):
            from sbdapm_env import SBDAPMEnv
            return SBDAPMEnv(**config)

    register_env("sbdapm_env", env_creator)


def export_onnx(algo, output_path):
    """Export trained policy to ONNX format."""
    try:
        import torch
        import torch.nn as nn

        # Get policy
        policy = algo.get_policy()
        model = policy.model

        # Create wrapper for ONNX export
        class PolicyWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, obs):
                # RLlib models expect dict input
                model_out, _ = self.model({"obs": obs})
                # Apply softmax for action probabilities
                action_probs = torch.softmax(model_out, dim=-1)
                return action_probs

        wrapper = PolicyWrapper(model)
        wrapper.eval()

        # Dummy input
        dummy_input = torch.randn(1, 71)

        # Export
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            input_names=["observation"],
            output_names=["action_probabilities"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action_probabilities": {0: "batch_size"}
            },
            opset_version=11
        )

        print(f"ONNX model exported to: {output_path}")
        return True

    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("Saving checkpoint instead...")
        return False


def train(args):
    """Main training loop."""
    print("=" * 60)
    print("SBDAPM RLlib Training")
    print("=" * 60)

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register environment
    register_env()

    # Create config and algorithm
    config = create_ppo_config()
    algo = config.build()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(SBDAPMConfig.OUTPUT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"Training for {args.iterations} iterations\n")

    # Training loop
    best_reward = float("-inf")

    for i in range(args.iterations):
        result = algo.train()

        # Extract metrics
        episode_reward_mean = result.get("episode_reward_mean", 0)
        episode_len_mean = result.get("episode_len_mean", 0)

        print(f"Iteration {i+1:4d}: "
              f"reward={episode_reward_mean:8.2f}, "
              f"len={episode_len_mean:6.1f}")

        # Save checkpoint
        if (i + 1) % args.checkpoint_freq == 0:
            checkpoint_path = algo.save(output_dir)
            print(f"  Checkpoint: {checkpoint_path}")

        # Track best
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
            best_checkpoint = algo.save(os.path.join(output_dir, "best"))
            print(f"  New best! reward={best_reward:.2f}")

    # Final save
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    final_checkpoint = algo.save(output_dir)
    print(f"Final checkpoint: {final_checkpoint}")

    # Export ONNX
    onnx_path = os.path.join(output_dir, f"{SBDAPMConfig.MODEL_NAME}.onnx")
    if export_onnx(algo, onnx_path):
        print(f"\nONNX model: {onnx_path}")
        print("\nTo use in Unreal Engine:")
        print(f'  TacticalPolicy->LoadPolicy(TEXT("{onnx_path}"))')

    # Cleanup
    algo.stop()
    ray.shutdown()

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train SBDAPM tactical policy with RLlib")
    parser.add_argument("--iterations", type=int, default=SBDAPMConfig.NUM_ITERATIONS,
                        help="Number of training iterations")
    parser.add_argument("--checkpoint-freq", type=int, default=SBDAPMConfig.CHECKPOINT_FREQ,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--host", type=str, default=SBDAPMConfig.HOST,
                        help="Schola gRPC server host")
    parser.add_argument("--port", type=int, default=SBDAPMConfig.PORT,
                        help="Schola gRPC server port")

    args = parser.parse_args()

    # Update config
    SBDAPMConfig.HOST = args.host
    SBDAPMConfig.PORT = args.port

    if not RLLIB_AVAILABLE:
        print("\nError: ray[rllib] is required. Install with:")
        print("  pip install ray[rllib] torch")
        sys.exit(1)

    if not SCHOLA_AVAILABLE:
        print("\nWarning: schola not installed. Using dummy environment.")
        print("For real training, install with:")
        print("  pip install schola[rllib]")

    train(args)


if __name__ == "__main__":
    main()
