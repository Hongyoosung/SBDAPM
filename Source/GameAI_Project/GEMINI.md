# SBDAPM: AlphaZero-Inspired Multi-Agent Combat AI

**Engine:** Unreal Engine 5.6 | **Language:** C++17 | **Platform:** Windows

---

## Architecture Overview

**Design Goal:** Real-time multi-agent combat system with coupled MCTS+RL, learned value functions, and world model simulation.

```
Team Leader (per team)
  ↓ Continuous MCTS (value network guided, 1-2s intervals)
  ↓ World Model (predicts 5-10 steps ahead)
  ↓ Strategic Commands (with confidence estimates)
  ↓
Followers (N agents)
  ↓ RL Policy Network (provides MCTS priors + tactical actions)
  ↓ State Tree Execution (command-driven states)
  ↓ Feedback Loop (RL experiences → MCTS curriculum)
```

**Key Innovations:**
- **Value Network**: Learned team state evaluation (replaces heuristics)
- **World Model**: State transition predictor (enables true Monte Carlo simulation)
- **Coupled Training**: MCTS guides RL curriculum, RL provides MCTS priors
- **Unified Rewards**: Hierarchical rewards align strategic + tactical objectives
- **Continuous Planning**: Proactive planning (1-2s intervals) with uncertainty quantification

---

## Core Components

### 1. Team Leader (`Team/TeamLeaderComponent.h/cpp`)
- **Continuous MCTS**: Runs at 1-2s intervals, proactive planning.
- **Value Network Guided**: Uses learned value function for tree search.
- **World Model Rollouts**: Predicts 5-10 steps ahead.
- **Strategic Commands**: Includes confidence estimates (visit count, value variance, entropy).
- **Exports MCTS Statistics**: Used for RL curriculum.

**Files:**
- `Team/TeamLeaderComponent.h/cpp`
- `AI/MCTS/MCTS.h/cpp`
- `AI/MCTS/TeamMCTSNode.h`

### 2. Value Network (`RL/TeamValueNetwork.h/cpp`)
**Purpose:** Estimate team state value to guide MCTS tree search.

**Architecture:**
```
Input: FTeamObservation (40 team + N×71 individual features)
  ↓ Embedding Layer (256 neurons, ReLU)
  ↓ Shared Trunk (256→256→128, ReLU)
  ↓ Value Head (128→64→1, Tanh)
Output: Team state value [-1, 1] (loss → win probability)
```

**Integration:**
- `MCTS.cpp:SimulateNode()` - Query value network for leaf evaluation
- `TeamLeaderComponent.cpp` - Load ONNX model via NNE

### 3. World Model (`Simulation/WorldModel.h/cpp`)
**Purpose:** Predict future states for true Monte Carlo simulation.

**Architecture:**
```
Input: CurrentState (TeamObs) + AllActions (commands + tactical actions)
  ↓ Action Encoder (commands → embeddings)
  ↓ State Encoder (observations → embeddings)
  ↓ Fusion Layer (concat + MLP)
  ↓ Transition Predictor (outputs state deltas)
Output: NextState (predicted TeamObs)
```

**Predictions:**
- Health changes, Position changes, Status effects.
- Stochastic sampling for uncertainty.

**Integration:**
- `MCTS.cpp:SimulateNode()` - Rollout 5-10 steps via WorldModel
- `TeamObservation.h` - Add `ApplyDelta(FStateTransition)` method

### 4. RL Policy Network (`RL/RLPolicyNetwork.h/cpp`)
- **Dual-head architecture**: Policy head + Prior head.
- **Policy Head**: Softmax probabilities for immediate action selection.
- **Prior Head**: Logits for MCTS node initialization (guide tree search).
- **Unified rewards**: Individual + coordination bonuses.
- **MCTS-guided curriculum**: Prioritized replay on high-uncertainty scenarios.

**Rewards:**
```cpp
// Individual
+10  Kill, +5 Damage, -5 Take Damage, -10 Death

// Coordination bonuses
+15  Kill while executing strategic command
+10  Coordinated action (combined fire, cover)
+5   Formation maintenance
-15  Disobey strategic command

// Strategic (team-level)
+50  Objective captured
+30  Enemy squad wiped
-30  Own squad wiped
```

**Files:**
- `RL/RLPolicyNetwork.h/cpp`
- `RL/HybridPolicyNetwork.h/cpp`
- `RL/RewardCalculator.h/cpp`
- `RL/CurriculumManager.h/cpp`

### 5. Followers (`Team/FollowerAgentComponent.h/cpp`)
- **Confidence-weighted execution**: Low confidence commands → RL can override.
- **Coordination tracking**: Detect combined actions with allies.
- **Experience tagging**: Mark experiences with MCTS uncertainty.

### 6. EQS Cover System (`EQS/*`)
- **Generator:** `EnvQueryGenerator_CoverPoints`
- **Test:** `EnvQueryTest_CoverQuality`
- **Context:** `EnvQueryContext_CoverEnemies`

### 7. Observations (`Observation/ObservationElement.h/cpp`, `TeamObservation.h/cpp`)
- **Features**: 71 individual + 40 team features.
- **World Model Support**: `ApplyDelta`, `Clone`, `Serialize`.

### 8. Simulation Manager (`Core/SimulationManagerGameMode.h/cpp`)
- Team registration and management.
- Enemy relationship tracking.
- Actor-to-team mapping.

### 9. Communication (`Team/TeamCommunicationManager.h/cpp`)
- Leader ↔ Follower message passing.
- Event priority system.
- MCTS statistics export.
- Coordination event tracking.

---

## Implementation Status

### Implemented & Validated
- Command Pipeline (Perception → MCTS → Commands → StateTree)
- MCTS tree search (Event-driven base)
- StateTree execution (Tasks, Evaluators, Conditions)
- Combat system (Health, Weapon, Rewards)
- Perception system
- Observations
- EQS cover system
- Simulation Manager GameMode

### In Progress
**Value Network + World Model**
- [ ] Implement `TeamValueNetwork.h/cpp`
- [ ] Implement `WorldModel.h/cpp`
- [ ] Modify `MCTS.cpp:SimulateNode()` to use both
- [ ] Create training scripts

**Coupled Training**
- [ ] MCTS → RL: Curriculum Manager
- [ ] RL → MCTS: Policy priors
- [ ] Dual-head `HybridPolicyNetwork`

**Rewards + Planning**
- [ ] Unified hierarchical rewards
- [ ] UCB action sampling
- [ ] Continuous planning (1-2s intervals)
- [ ] Confidence estimates in commands

**Self-Play Pipeline**
- [ ] `self_play_collector.py`
- [ ] End-to-end training loop
- [ ] Evaluate vs baseline

---

## Work Instructions

### Token Efficiency (CRITICAL)
1. **NO verbose reports** - Code-focused, concise documentation only
2. **NO long explanations** - Implement first, brief comments when needed
3. **NO redundant updates** - Don't repeat what's in this file
4. **Focus on implementation** - Spend tokens on code, not prose

### Code Style
- Direct implementation over planning documents
- Use file:line references (e.g., `MCTS.cpp:73`)
- Minimal comments unless logic is complex

### Architecture Rules
- **Followers NEVER run MCTS** (only leader)
- **MCTS runs continuously** (1-2s intervals)
- **Value network guides MCTS** (replaces heuristics)
- **World model enables simulation** (predict 5-10 steps)
- **RL provides MCTS priors** (dual-head network)
- **MCTS guides RL curriculum** (prioritized replay on uncertainty)
- **Rewards are aligned** (individual + coordination + strategic)

### Performance Targets
- Team Leader MCTS: 30-50ms
- RL Inference: 1-3ms
- World Model Prediction: 5-10ms
- StateTree Tick: <0.5ms per agent
- Total Frame Budget: 10-20ms for 4-agent team

---

## File Structure

```
Source/GameAI_Project/
├── MCTS/
│   ├── MCTS.h/cpp                    # ValueNetwork + WorldModel integration
│   ├── TeamMCTSNode.h                # Add ActionPriors
│   └── CommandSynergy.h/cpp          # Synergy score computation
├── RL/
│   ├── RLPolicyNetwork.h/cpp         # Add GetActionPriors()
│   ├── TeamValueNetwork.h/cpp        # Team state value estimation
│   ├── HybridPolicyNetwork.h/cpp     # Dual-head (policy + priors)
│   ├── RewardCalculator.h/cpp        # Unified reward system
│   ├── CurriculumManager.h/cpp       # MCTS-guided training
│   └── RLReplayBuffer.h/cpp          # Experience storage
├── Simulation/
│   ├── WorldModel.h/cpp              # State transition predictor
│   └── StateTransition.h             # State delta structs
├── StateTree/
│   ├── Tasks/
│   ├── Evaluators/
│   ├── Conditions/
│   └── FollowerStateTreeComponent.h/cpp
├── Combat/
...
```
