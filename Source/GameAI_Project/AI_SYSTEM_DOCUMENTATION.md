# AI System Documentation: AlphaZero-Inspired Multi-Agent Combat AI

**Version:** 3.0
**Engine:** Unreal Engine 5.6
**Language:** C++17
**Last Updated:** 2025-11-26

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Components](#architecture-components)
3. [Team Leader System (Strategic MCTS)](#team-leader-system-strategic-mcts)
4. [Follower Agent System (Tactical RL)](#follower-agent-system-tactical-rl)
5. [Value Network](#value-network)
6. [World Model](#world-model)
7. [Coupled Training](#coupled-training)
8. [Reward System](#reward-system)
9. [Observation System](#observation-system)
10. [Training Pipeline](#training-pipeline)
11. [Performance Characteristics](#performance-characteristics)
12. [Configuration](#configuration)
13. [Troubleshooting](#troubleshooting)

---

## System Overview

### Design Philosophy

The AI system is inspired by AlphaZero but adapted for **real-time, multi-agent, partial observability** combat scenarios. It combines:

- **Strategic Planning**: MCTS tree search for team-level decisions
- **Tactical Execution**: Deep RL for individual agent actions
- **Learned Value Functions**: Neural networks replace hand-crafted heuristics
- **World Model Simulation**: State transition prediction for Monte Carlo rollouts
- **Coupled Training**: MCTS guides RL curriculum, RL provides MCTS priors

### Key Innovations (v3.0)

1. **Value Network**: Estimates team state value → guides MCTS tree search
2. **World Model**: Predicts future states → enables true simulation
3. **Prior-Guided MCTS**: RL policy provides action priors → focuses search
4. **MCTS-Guided Curriculum**: High-uncertainty scenarios prioritized in RL training
5. **Hierarchical Rewards**: Unified reward system aligns strategic + tactical objectives
6. **Continuous Planning**: Proactive MCTS execution (1-2s intervals) with uncertainty quantification
7. **Automated Self-Play**: Complete training pipeline from data collection to evaluation

### System Hierarchy

```
SimulationManagerGameMode (manages teams, enemy relationships)
    ↓
TeamLeaderComponent (per team) - Strategic Layer
    ├─ MCTS Tree Search (continuous, 1.5s intervals)
    │   ├─ ValueNetwork (leaf evaluation)
    │   ├─ WorldModel (state prediction)
    │   └─ RLPolicyNetwork priors (action initialization)
    ├─ Command Generation (strategic commands with confidence)
    ├─ Curriculum Manager (identifies hard scenarios)
    └─ Performance Profiling (tracks MCTS timing)
    ↓
FollowerAgentComponent (per agent) - Tactical Layer
    ├─ Perception (enemy detection, environment raycasting)
    ├─ RLPolicyNetwork (atomic action selection - 8D continuous)
    ├─ StateTree (execution framework)
    │   ├─ Task (ExecuteObjective - unified objective-based execution)
    │   ├─ Evaluators (UpdateObservation, SpatialContext)
    │   └─ Conditions (IsAlive, CheckObjectiveType)
    ├─ Combat Systems (Health, Weapon)
    └─ Reward Calculator (hierarchical rewards)
```

---

## Architecture Components

### Core Files

#### Strategic Layer
- **Team/TeamLeaderComponent.h/cpp** (TeamLeaderComponent.cpp:1-800+)
  - Continuous MCTS execution (1.5s configurable intervals)
  - Strategic command generation with uncertainty
  - Curriculum scenario tracking
  - Performance profiling

- **AI/MCTS/MCTS.h/cpp** (MCTS.cpp:1-600+)
  - AlphaZero-style tree search with PUCT
  - Value network integration for leaf evaluation
  - World model integration for multi-step rollouts
  - Prior-guided node expansion
  - UCB-based action sampling

- **AI/MCTS/TeamMCTSNode.h** (TeamMCTSNode.h:1-120)
  - Node structure with ActionPriors
  - PUCT calculation for exploration-exploitation balance
  - Visit count and value tracking

#### Tactical Layer
- **Team/FollowerAgentComponent.h/cpp** (FollowerAgentComponent.cpp:1-900+)
  - Observation building (71 individual + 40 team features)
  - RL policy integration
  - Combat event handling
  - Confidence-weighted command execution
  - State transition logging

- **RL/RLPolicyNetwork.h/cpp** (RLPolicyNetwork.h:1-200, RLPolicyNetwork.cpp:1-500+)
  - 3-layer MLP (128→128→64 neurons)
  - 16 tactical actions (Move, Assault, Defend, TakeCover, etc.)
  - Objective-based priors for MCTS
  - Experience storage with MCTS uncertainty tagging
  - ONNX inference via NNE

#### Neural Networks
- **RL/TeamValueNetwork.h/cpp** (TeamValueNetwork.cpp:1-400+)
  - Input: 40 team + N×71 individual features
  - Architecture: Embedding(256) → Trunk(256→256→128) → Value(128→64→1)
  - Output: Team state value [-1, 1] (loss → win probability)
  - Training: TD-learning on MCTS outcomes

- **Simulation/WorldModel.h/cpp** (WorldModel.cpp:1-500+)
  - Input: TeamObservation + All actions (strategic + tactical)
  - Architecture: ActionEncoder → StateEncoder → Fusion → Transition Predictor
  - Output: Predicted next state (FStateTransition)
  - Training: Supervised learning on real transitions
  - Integration: 5-step rollouts in MCTS simulation

- **RL/HybridPolicyNetwork.h/cpp** (HybridPolicyNetwork.cpp:1-300)
  - Dual-head architecture
  - Policy Head: Softmax probabilities (immediate action selection)
  - Prior Head: Logits (MCTS node initialization)
  - Training: PPO with MCTS prior targets

#### Training Support
- **RL/CurriculumManager.h/cpp** (CurriculumManager.cpp:1-300)
  - Tracks MCTS high-uncertainty scenarios
  - Prioritizes hard situations for RL training
  - Configurable sampling strategies

- **RL/RewardCalculator.h/cpp** (RewardCalculator.cpp:1-400+)
  - Hierarchical reward computation
  - Individual: +10 kill, +5 damage, -5 take damage, -10 death
  - Coordination: +15 strategic kill, +10 combined fire, +5 formation
  - Strategic: +50 objective, +30 team wipe
  - Tracks formation adherence, objective compliance

---

## Team Leader System (Strategic MCTS)

### Continuous Planning

**File:** TeamLeaderComponent.cpp:200-250

**Key Features:**
- Proactive planning every 1-2s (configurable)
- Critical events can interrupt (priority threshold)
- Async execution on background thread
- Performance profiling with rolling averages

### MCTS Tree Search

**File:** MCTS.cpp:100-400

#### Selection (PUCT with Priors)

#### Expansion (Prior-Guided)

**File:** MCTS.cpp:250-300

**UCB Action Sampling:**
- Top-3 objectives per follower (based on distance, threat, strategic value)
- Synergy bonuses for complementary actions
- Epsilon-greedy exploration (20%)
- Progressive widening as visit count increases

#### Simulation (World Model Rollouts)

**File:** MCTS.cpp:350-400

#### Backpropagation

**File:** MCTS.cpp:420-450

### Command Generation

**File:** TeamLeaderComponent.cpp:400-500

### Uncertainty Quantification

**File:** TeamLeaderComponent.cpp:550-600

## Follower Agent System (Tactical RL)

### RL Policy Network Integration

**File:** FollowerAgentComponent.cpp:200-300

### Confidence-Weighted Command Execution

**File:** FollowerAgentComponent.cpp:350-400

### StateTree Execution (v3.0)

**File:** StateTree/FollowerStateTreeComponent.cpp:50-150

StateTree is the **PRIMARY** execution system. It:
- Drives state transitions (Idle, Active, Dead)
- Executes atomic actions via unified ExecuteObjective task
- Syncs with objectives from team leader
- Checks conditions for state transitions

**Key Tasks (v3.0):**
- `STTask_ExecuteObjective` - Unified objective-based execution
  - Queries RL network for atomic actions (8D: move, aim, fire, crouch, ability)
  - Executes movement, aiming, and discrete actions
  - Calculates hierarchical rewards (individual + coordination + strategic)
  - Handles ALL objective types (Eliminate, Capture, Defend, Support, Rescue)

**Key Evaluators:**
- `STEvaluator_UpdateObservation` - Updates observation data
- `STEvaluator_SpatialContext` - Updates action space mask for valid movement

**Key Conditions:**
- `STCondition_CheckObjectiveType` - Checks current objective type
- `STCondition_IsAlive` - Checks if agent is alive

### Experience Storage

**File:** FollowerAgentComponent.cpp:700-750

## Value Network

**File:** RL/TeamValueNetwork.cpp:1-400

### Architecture

```
Input: FTeamObservation (40 team + N×71 individual features)
  ↓
Embedding Layer (Linear: InputDim → 256, ReLU)
  ↓
Shared Trunk
  ├─ Layer 1: Linear(256 → 256, ReLU)
  ├─ Layer 2: Linear(256 → 128, ReLU)
  ↓
Value Head
  ├─ Layer 1: Linear(128 → 64, ReLU)
  └─ Layer 2: Linear(64 → 1, Tanh)
  ↓
Output: Team state value [-1, 1] (loss → win probability)
```

### Inference (C++)

### Training (Python)

**File:** Scripts/train_value_network.py:1-300

## World Model

**File:** Simulation/WorldModel.cpp:1-500

### Architecture

```
Input: CurrentState (TeamObs) + AllActions (strategic + tactical)
  ↓
Action Encoder (commands → embeddings, 64-dim per action)
  ↓
State Encoder (observations → embeddings, 256-dim)
  ↓
Fusion Layer (Concatenate + MLP: 320 → 256 → 256)
  ↓
Transition Predictor
  ├─ Health Delta Head (256 → 128 → N_agents)
  ├─ Position Delta Head (256 → 128 → N_agents×3)
  ├─ Status Effect Head (256 → 128 → N_agents×StatusDim)
  └─ Terminal Prediction (256 → 64 → 1, Sigmoid)
  ↓
Output: FStateTransition (deltas + terminal flag + reward)
```

### Inference (C++)

### Training (Python)

**File:** Scripts/train_world_model.py:1-400

## Coupled Training

### MCTS → RL: Curriculum Manager

**File:** RL/CurriculumManager.cpp:1-300

### RL → MCTS: Policy Priors

**File:** RL/RLPolicyNetwork.cpp:400-500
---

## Reward System

**File:** RL/RewardCalculator.cpp:1-400

### Hierarchical Reward Structure



### Coordination Detection


## Observation System

**File:** Observation/ObservationElement.cpp:1-400, TeamObservation.cpp:1-300

### Individual Observation (71 features)

### Team Observation (40 features)
---

## Training Pipeline

### Self-Play Data Collection

**Script:** Scripts/self_play_collector.py

```bash
# Collect 1000 games of self-play data
python self_play_collector.py --games 1000 --output ./selfplay_data --save-interval 50

# Data collected:
# - RL experiences (observations, actions, rewards, next_obs)
# - MCTS traces (team_obs, commands, visit_counts, final_outcome)
# - State transitions (current_state, actions, next_state)
# - Game outcomes (win/loss/draw, kills, deaths, coordination metrics)
```

### Coupled Training

**Script:** Scripts/train_coupled_system.py

```bash
# Train all networks on collected data
python train_coupled_system.py \
    --data-dir ./selfplay_data \
    --output-dir ./training_output \
    --iterations 1 \
    --value-epochs 50 \
    --world-epochs 50 \
    --rl-epochs 50 \
    --batch-size 64 \
    --copy-to-ue5 ../Content/AI/Models

# Trains:
# 1. ValueNetwork on MCTS outcomes (TD-learning)
# 2. WorldModel on state transitions (supervised)
# 3. RLPolicy on RL experiences (PPO with prioritized replay)
#
# Exports to ONNX and copies to UE5 project
```

### Evaluation

**Script:** Scripts/evaluate_agents.py

```bash
# Compare v3.0 against v2.0 baseline
python evaluate_agents.py \
    --data ./evaluation_data \
    --output ./evaluation_results \
    --baseline v2.0 \
    --trained v3.0 \
    --plots

# Metrics:
# - Win rate (target: ≥70%)
# - K/D ratio
# - Coordination rate (target: ≥30%)
# - MCTS efficiency (target: 50% fewer simulations)
# - Damage efficiency
```

### Complete Pipeline

**Script:** Scripts/run_selfplay_pipeline.py

```bash
# Run complete training loop (1000 games over 10 iterations)
python run_selfplay_pipeline.py \
    --games 100 \
    --iterations 10 \
    --output ./pipeline_output \
    --ue5-project "C:/Projects/SBDAPM"

# Pipeline steps (per iteration):
# 1. Collect 100 games of self-play data
# 2. Train all networks
# 3. Export to ONNX and deploy to UE5
# 4. Evaluate (every 2 iterations)
# 5. Repeat
```

---

## Performance Characteristics

### Timing Targets (v3.0)

| Component | Target | Current (v2.0) | Notes |
|-----------|--------|----------------|-------|
| MCTS Tree Search | 30-50ms | ~34ms | Improved with value network pruning |
| RL Inference | 1-3ms | ~2ms | NNE optimized, GPU inference |
| World Model Prediction | 5-10ms | N/A | 5-step rollout |
| Value Network Inference | 2-5ms | N/A | Single forward pass |
| StateTree Tick | <0.5ms | ~0.3ms | Per agent |
| **Total Frame Budget** | **10-20ms** | **~15ms** | For 4-agent team |

### Memory Usage

- MCTS Tree: ~5-10 MB (1000 simulations, 10 depth)
- Value Network: ~2 MB (ONNX model)
- World Model: ~5 MB (ONNX model)
- RL Policy: ~1 MB (ONNX model)
- Observations: ~50 KB per agent

### Scalability

- Tested with **4-8 agents per team**
- MCTS scales linearly with team size
- RL inference per agent (independent)
- World model scales with state size (quadratic in agent count)

---

## Configuration

### MCTS Parameters

**File:** AI/MCTS/MCTS.h:30-50

```cpp
UPROPERTY(EditAnywhere, Category = "MCTS")
int32 MaxSimulations = 1000;  // Simulations per MCTS run

UPROPERTY(EditAnywhere, Category = "MCTS")
int32 SearchDepth = 10;  // Max tree depth

UPROPERTY(EditAnywhere, Category = "MCTS")
float ExplorationConstant = 1.41f;  // UCB exploration (sqrt(2))

UPROPERTY(EditAnywhere, Category = "MCTS")
int32 MaxCombinations = 10;  // Action combinations to expand

UPROPERTY(EditAnywhere, Category = "MCTS")
float EpsilonGreedy = 0.2f;  // Exploration probability
```

### Continuous Planning

**File:** Team/TeamLeaderComponent.h:80-90

```cpp
UPROPERTY(EditAnywhere, Category = "Continuous Planning")
bool bEnableContinuousPlanning = true;

UPROPERTY(EditAnywhere, Category = "Continuous Planning")
float ContinuousPlanningInterval = 1.5f;  // Seconds between MCTS runs

UPROPERTY(EditAnywhere, Category = "Continuous Planning")
int32 CriticalEventPriority = 9;  // Priority threshold to interrupt planning
```

### Confidence Thresholds

**File:** Team/FollowerAgentComponent.h:100-110

```cpp
UPROPERTY(EditAnywhere, Category = "Command Execution")
float ConfidenceThreshold = 0.5f;  // Minimum confidence to execute command

UPROPERTY(EditAnywhere, Category = "Command Execution")
float HighEntropyThreshold = 1.5f;  // Threshold for high uncertainty warning

UPROPERTY(EditAnywhere, Category = "Command Execution")
bool bAllowTacticalOverride = true;  // Allow RL to override low-confidence commands
```

### Reward Weights

**File:** RL/RewardCalculator.h:40-70

```cpp
// Individual rewards
UPROPERTY(EditAnywhere, Category = "Rewards|Individual")
float IndividualReward_Kill = 10.0f;

UPROPERTY(EditAnywhere, Category = "Rewards|Individual")
float IndividualReward_Damage = 5.0f;

UPROPERTY(EditAnywhere, Category = "Rewards|Individual")
float IndividualReward_TakeDamage = -5.0f;

UPROPERTY(EditAnywhere, Category = "Rewards|Individual")
float IndividualReward_Death = -10.0f;

// Coordination bonuses
UPROPERTY(EditAnywhere, Category = "Rewards|Coordination")
float CoordinationBonus_StrategicKill = 15.0f;

UPROPERTY(EditAnywhere, Category = "Rewards|Coordination")
float CoordinationBonus_CombinedFire = 10.0f;

UPROPERTY(EditAnywhere, Category = "Rewards|Coordination")
float CoordinationBonus_Formation = 5.0f;

UPROPERTY(EditAnywhere, Category = "Rewards|Coordination")
float CoordinationPenalty_Disobey = -15.0f;

// Strategic rewards
UPROPERTY(EditAnywhere, Category = "Rewards|Strategic")
float StrategicReward_Objective = 50.0f;

UPROPERTY(EditAnywhere, Category = "Rewards|Strategic")
float StrategicReward_EnemyWipe = 30.0f;

UPROPERTY(EditAnywhere, Category = "Rewards|Strategic")
float StrategicPenalty_OwnWipe = -30.0f;
```

---

## Troubleshooting

### MCTS Performance Issues

**Symptom:** MCTS taking >100ms

**Causes:**
1. Too many simulations (MaxSimulations > 1500)
2. Too many action combinations (MaxCombinations > 20)
3. Deep search depth (SearchDepth > 15)
4. Value network inference slow

**Solutions:**
- Reduce MaxSimulations to 500-1000
- Limit MaxCombinations to 10-15
- Reduce SearchDepth to 8-10
- Ensure ONNX models are optimized (quantized)
- Check GPU inference is enabled (NNE settings)

### Value Network Not Loading

**Symptom:** "ValueNetwork: No ONNX model loaded" warnings

**Causes:**
1. ONNX file not in correct directory
2. Model name mismatch
3. NNE plugin not enabled

**Solutions:**
- Verify ONNX file in `Content/AI/Models/value_network_latest.onnx`
- Check model path in TeamLeaderComponent properties
- Enable NNE plugin in Project Settings

### Low Confidence Commands

**Symptom:** Frequent "Low confidence command" warnings

**Causes:**
1. Insufficient MCTS simulations
2. High policy entropy (ambiguous situations)
3. Value network not trained

**Solutions:**
- Increase MaxSimulations
- Train value network on more data
- Lower ConfidenceThreshold (but may reduce quality)
- Check world model predictions are accurate

### Coordination Not Detected

**Symptom:** Low coordination bonus despite visual coordination

**Causes:**
1. CombinedFireWindow too narrow
2. Formation tolerance too strict
3. Timing issues (actions not synchronized)

**Solutions:**
- Increase CombinedFireWindow from 2.0s to 3.0s
- Increase DistanceTolerance from 500 to 800
- Check command sync timing in TeamCommunicationManager

### Training Not Converging

**Symptom:** Loss plateaus, no improvement

**Causes:**
1. Learning rate too high/low
2. Insufficient data diversity
3. Reward scaling issues
4. Overfitting

**Solutions:**
- Adjust learning rates (0.0001-0.001 for value/world, 0.0003 for RL)
- Collect more diverse self-play data
- Normalize rewards (divide by max episode reward)
- Add dropout (0.2-0.3) and L2 regularization

---

## Next Steps

1. **Data Collection**: Run 1000+ self-play games in UE5 with data export enabled
2. **Training**: Execute `python run_selfplay_pipeline.py --games 1000 --iterations 10`
3. **Evaluation**: Compare v3.0 against v2.0 baseline
4. **Iteration**: Repeat training cycle based on evaluation results
5. **Deployment**: Use trained models for production gameplay

---

**Document Version:** 1.0
**Last Updated:** 2025-11-26
**Maintained By:** SBDAPM Team
