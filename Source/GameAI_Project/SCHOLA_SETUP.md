# Schola RL Training Setup Guide

## Overview

The Schola integration has been fully implemented for real-time PPO training of follower agents. The system properly manages the gRPC server connection between UE5 and Python RLlib.

## Architecture

```
ScholaCombatEnvironment (Actor)
  ├─ Manages gRPC server (port 50051)
  ├─ Auto-discovers ScholaAgentComponents
  ├─ Creates FollowerAgentTrainer for each agent
  └─ Integrates with SimulationManagerGameMode episodes
      │
      ├─ FollowerAgentTrainer (per agent)
      │   ├─ Wraps ScholaAgentComponent
      │   ├─ Provides rewards (TacticalRewardProvider)
      │   └─ Tracks episode termination
      │
      └─ ScholaAgentComponent (on each follower pawn)
          ├─ TacticalObserver (71 features)
          ├─ TacticalRewardProvider (combat rewards)
          └─ TacticalActuator (8D actions)
```

## Setup Instructions

### 1. Level Setup

**Add ScholaCombatEnvironment to your level:**

1. Open your combat level in UE5
2. Add Actor → Place Actor → `ScholaCombatEnvironment`
3. Configure in Details panel:
   - `bEnableTraining` = **true** (starts gRPC server)
   - `ServerPort` = **50051** (default)
   - `bAutoDiscoverAgents` = **true** (finds all agents automatically)
   - `TrainingTeamIDs` = **[]** (empty = train all teams, or specify team IDs)

**Ensure Follower Pawns have ScholaAgentComponent:**

1. Open your follower pawn Blueprint
2. Add Component → `ScholaAgentComponent`
3. The component auto-configures:
   - Finds `FollowerAgentComponent` automatically
   - Creates `TacticalObserver`, `TacticalRewardProvider`, `TacticalActuator`
   - Registers with `ScholaCombatEnvironment` on BeginPlay

### 2. Episode Management

The integration automatically syncs with `SimulationManagerGameMode` episodes:

**When Episode Starts:**
- `ScholaCombatEnvironment::OnEpisodeStarted()` is called
- All agents are reset via `ScholaAgentComponent::ResetEpisode()`
- Schola environment calls `Reset()` to notify Python

**When Episode Ends:**
- `SimulationManagerGameMode` detects team elimination or timeout
- `ScholaCombatEnvironment::OnEpisodeEnded()` is called
- Schola environment calls `MarkCompleted()` to trigger final experience collection

### 3. Python Training Script

**Start UE5 first, then run Python:**

```bash
cd Scripts
python train_rllib.py
```

**The Python script (`train_rllib.py`) should:**
1. Connect to `localhost:50051` (gRPC client)
2. Define the environment using `sbdapm_env.py`
3. Configure RLlib PPO trainer
4. Start training loop

**Example Python setup:**

```python
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from sbdapm_env import SBDAPMEnv

ray.init()

config = (
    PPOConfig()
    .environment(SBDAPMEnv, env_config={"server": "localhost:50051"})
    .framework("torch")
    .training(
        lr=3e-4,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        entropy_coeff=0.01,
        train_batch_size=4000,
    )
    .rollouts(num_rollout_workers=1)
)

algo = config.build()

for i in range(1000):
    result = algo.train()
    print(f"Episode {i}: reward={result['episode_reward_mean']}")

    # Save checkpoint every 100 iterations
    if i % 100 == 0:
        checkpoint = algo.save()
        print(f"Checkpoint saved at {checkpoint}")
```

### 4. Logs & Debugging

**UE5 Logs:**

```
LogTemp: [ScholaEnv] Initialized with 8 agents (Training: ON, Port: 50051)
LogTemp: [ScholaEnv] ✓ gRPC server started on port 50051
LogTemp: [ScholaEnv] ✓ Ready for Python RLlib connection
LogTemp: [ScholaEnv] Registered 8 trainers
LogTemp: [ScholaEnv] Episode 1 started - Resetting agents
LogTemp: [FollowerTrainer] Follower_BP_FollowerPawn_0 - Reset for new episode
```

**Python Logs:**

```
INFO: Connected to UE5 gRPC server at localhost:50051
INFO: Environment initialized with 8 agents
INFO: Episode 1: reward_mean=45.3, len_mean=234
```

### 5. Troubleshooting

**Issue: "CommunicationManager subsystem not found"**
- **Fix:** Ensure Schola plugin is enabled in `.uproject`
- Check: `Plugins` → `Schola` → ☑ Enabled
- Restart UE5 after enabling

**Issue: "No agents discovered"**
- **Fix:** Ensure follower pawns have `ScholaAgentComponent` attached
- Check: `ScholaCombatEnvironment.bAutoDiscoverAgents = true`
- Verify: Agents are spawned before environment initializes

**Issue: "Python can't connect to gRPC server"**
- **Fix:** Start UE5 first, wait for log: `✓ gRPC server started on port 50051`
- Check firewall: Allow port 50051
- Verify: `ScholaCombatEnvironment.bEnableTraining = true`

**Issue: "Episode never ends"**
- **Fix:** Check `SimulationManagerGameMode.bAutoRestartEpisode = true`
- Verify: Agents have `HealthComponent` (for death detection)
- Check: `MaxStepsPerEpisode` is set (or 0 for unlimited)

### 6. Key Configuration Options

**ScholaCombatEnvironment:**
- `bEnableTraining` - Start gRPC server (true for training, false for inference)
- `ServerPort` - gRPC port (default: 50051)
- `bAutoDiscoverAgents` - Auto-find ScholaAgentComponents (recommended: true)
- `TrainingTeamIDs` - Filter teams for training (empty = all teams)

**SimulationManagerGameMode:**
- `bAutoStartSimulation` - Start simulation on BeginPlay (recommended: true)
- `bAutoRestartEpisode` - Auto-restart after episode ends (recommended: true)
- `EpisodeRestartDelay` - Delay before restart (default: 2s)
- `MaxStepsPerEpisode` - Max steps before timeout (0 = unlimited)

**FollowerAgentTrainer:**
- `DecisionFrequency` - Steps between actions (default: 1 = every step)
- `MaxSteps` - Episode timeout threshold (default: 10000 steps)

## File Locations

**New Files:**
```
Public/Schola/ScholaCombatEnvironment.h
Private/Schola/ScholaCombatEnvironment.cpp
Public/Schola/FollowerAgentTrainer.h
Private/Schola/FollowerAgentTrainer.cpp
```

**Modified Files:**
```
Public/Schola/ScholaAgentComponent.h  (removed unused bEnableScholaTraining)
Private/Schola/ScholaAgentComponent.cpp  (removed gRPC server logic)
```

## Training Workflow

1. **Setup Level**
   - Place `ScholaCombatEnvironment` in level
   - Ensure follower pawns have `ScholaAgentComponent`

2. **Start UE5**
   - Press Play
   - Wait for log: `✓ gRPC server started on port 50051`

3. **Start Python**
   - Run `python Scripts/train_rllib.py`
   - Python connects to UE5 via gRPC
   - Training begins automatically

4. **Monitor Training**
   - UE5 logs: Episode progress, rewards, termination
   - Python logs: Mean reward, episode length, training metrics
   - TensorBoard: `tensorboard --logdir=~/ray_results`

5. **Export Model**
   - RLlib saves checkpoints every N iterations
   - Export to ONNX: `export_onnx.py checkpoint_path`
   - Load in UE5: `RLPolicyNetwork.ModelPath = "rl_policy_network.onnx"`

## Next Steps

1. **Test the integration:**
   - Start UE5 with ScholaCombatEnvironment in level
   - Verify gRPC server starts successfully
   - Test Python connection (even without full training script)

2. **Implement Python environment:**
   - Update `Scripts/sbdapm_env.py` with correct observation/action spaces
   - Ensure gRPC client connects to UE5 server
   - Test observation collection and action execution

3. **Train first model:**
   - Run `train_rllib.py` for 100-1000 iterations
   - Monitor convergence (reward should increase)
   - Export ONNX model

4. **Deploy trained model:**
   - Load ONNX in `RLPolicyNetwork` component
   - Switch `ScholaCombatEnvironment.bEnableTraining = false`
   - Run inference-only gameplay

---

**Last Updated:** 2025-11-28 (v3.1 Real-Time Training)
