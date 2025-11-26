# SBDAPM: AlphaZero-Inspired Multi-Agent Combat AI

**Engine:** Unreal Engine 5.6 | **Language:** C++17 | **Platform:** Windows

---

## Architecture Overview

### v3.0 (Target - AlphaZero-Inspired)
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

**See `REFACTORING_PLAN.md` for implementation roadmap.**

---

### v2.0 (Current Implementation)
**Hierarchical Team System:** Leader (MCTS strategic) → Followers (RL tactical + StateTree execution)

```
Team Leader (per team) → Event-driven MCTS → Strategic commands
    ↓
Followers (N agents) → RL Policy + State Tree → Tactical execution
```

**Current Status:**
- ✅ MCTS tree search (~34ms, event-driven)
- ✅ RL policy structure (rule-based fallback, no trained model yet)
- ✅ StateTree execution system
- ✅ Perception + Combat + Observations integrated
- ⚠️ Using hand-crafted heuristics (no value network)
- ⚠️ Static evaluation (no world model simulation)
- ⚠️ Decoupled training (MCTS and RL independent)

---

## Core Components

### 1. Team Leader (`Team/TeamLeaderComponent.h/cpp`)
**Current (v2.0):**
- Event-driven MCTS (async, 500-1000 simulations)
- Hand-crafted reward heuristics
- Static evaluation (no future state prediction)
- Strategic commands: Assault, Defend, Move, Support, Retreat

**Target (v3.0):**
- Continuous MCTS (1-2s intervals, proactive)
- Value network guided tree search
- World model rollouts (5-10 step lookahead)
- Commands include confidence estimates (visit count, value variance, entropy)
- Exports MCTS statistics for RL curriculum

**Files:**
- `Team/TeamLeaderComponent.h/cpp`
- `AI/MCTS/MCTS.h/cpp`
- `AI/MCTS/TeamMCTSNode.h`

---

### 2. Value Network (NEW - v3.0)
**Purpose:** Estimate team state value to guide MCTS tree search (replaces `CalculateTeamReward()` heuristics)

**Architecture:**
```
Input: FTeamObservation (40 team + N×71 individual features)
  ↓ Embedding Layer (256 neurons, ReLU)
  ↓ Shared Trunk (256→256→128, ReLU)
  ↓ Value Head (128→64→1, Tanh)
Output: Team state value [-1, 1] (loss → win probability)
```

**Training:**
- TD-learning on MCTS rollout outcomes
- Self-play data collection
- Supervised learning on game results

**Integration:**
- `MCTS.cpp:SimulateNode()` - Query value network for leaf evaluation
- `TeamLeaderComponent.cpp` - Load ONNX model via NNE

**Files (NEW):**
- `RL/TeamValueNetwork.h/cpp`
- `Scripts/train_value_network.py`

---

### 3. World Model (NEW - v3.0)
**Purpose:** Predict future states for true Monte Carlo simulation (not just static evaluation)

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
- Health changes (combat damage model)
- Position changes (movement dynamics)
- Status effects (buffs, debuffs)
- Stochastic sampling for uncertainty

**Training:**
- Supervised learning on real transitions: (S_t, A_t, S_{t+1})
- MSE loss on state prediction
- Logged during gameplay

**Integration:**
- `MCTS.cpp:SimulateNode()` - Rollout 5-10 steps via WorldModel
- `TeamObservation.h` - Add `ApplyDelta(FStateTransition)` method

**Files (NEW):**
- `Simulation/WorldModel.h/cpp`
- `Simulation/StateTransition.h`
- `Scripts/train_world_model.py`

---

### 4. RL Policy Network (`RL/RLPolicyNetwork.h/cpp`)
**Current (v2.0):**
- 3-layer network (128→128→64 neurons)
- 16 tactical actions (`ETacticalAction` enum)
- Rule-based fallback (no trained model)
- PPO training (offline, Python)
- Rewards: +10 kill, +5 damage, -5 take damage, -10 die

**Target (v3.0):**
- **Dual-head architecture**: Policy head + Prior head
- **Policy Head**: Softmax probabilities for immediate action selection
- **Prior Head**: Logits for MCTS node initialization (guide tree search)
- **Unified rewards**: Individual + coordination bonuses (align with MCTS objectives)
- **MCTS-guided curriculum**: Prioritized replay on high-uncertainty scenarios

**New Rewards (v3.0):**
```cpp
// Individual (same)
+10  Kill, +5 Damage, -5 Take Damage, -10 Death

// Coordination bonuses (NEW)
+15  Kill while executing strategic command
+10  Coordinated action (combined fire, cover)
+5   Formation maintenance
-15  Disobey strategic command

// Strategic (team-level, MCTS)
+50  Objective captured
+30  Enemy squad wiped
-30  Own squad wiped
```

**Files:**
- `RL/RLPolicyNetwork.h/cpp` - Add `GetActionPriors()`
- `RL/HybridPolicyNetwork.h/cpp` (NEW) - Dual-head architecture
- `RL/RewardCalculator.h/cpp` (NEW) - Unified reward computation
- `RL/CurriculumManager.h/cpp` (NEW) - MCTS-guided experience prioritization

---

### 5. Followers (`Team/FollowerAgentComponent.h/cpp`)
**Current (v2.0):**
- Receives strategic commands from leader
- RL policy selects tactical actions
- Signals events to leader (enemy spotted, ally killed)
- Integrates with StateTree for execution

**Target (v3.0):**
- **Confidence-weighted execution**: Low confidence commands → RL can override with tactical judgment
- **Coordination tracking**: Detect combined actions with allies (bonus rewards)
- **Experience tagging**: Mark experiences with MCTS uncertainty (prioritized replay)

**Files:**
- `Team/FollowerAgentComponent.h/cpp`
- `Team/StrategicCommand.h` - Add confidence fields (Confidence, ValueVariance, PolicyEntropy)



### 9. EQS Cover System (`EQS/*`)
**Status:** ✅ v2.0 Implemented (no changes needed for v3.0)

- **Generator:** `EnvQueryGenerator_CoverPoints` - Grid/tag-based cover generation
- **Test:** `EnvQueryTest_CoverQuality` - Multi-factor scoring
- **Context:** `EnvQueryContext_CoverEnemies` - Team Leader integration

---

### 10. Observations (`Observation/ObservationElement.h/cpp`, `TeamObservation.h/cpp`)
**Current:** ✅ 71 individual + 40 team features

**Target (v3.0):** Add methods for world model
- `TeamObservation::ApplyDelta(FStateTransition)` - Apply predicted state changes
- `ObservationElement::Clone()` - Deep copy for simulation
- `TeamObservation::Serialize()` - Export for world model training

---

### 11. Simulation Manager (`Core/SimulationManagerGameMode.h/cpp`)
**Status:** ✅ v2.0 Implemented (no changes needed for v3.0)

- Team registration and management
- Enemy relationship tracking (mutual enemies, FFA)
- Actor-to-team mapping (O(1) lookup)

---

### 12. Communication (`Team/TeamCommunicationManager.h/cpp`)
**Status:** ✅ v2.0 Implemented (minor changes for v3.0)

- Leader ↔ Follower message passing
- Event priority system (triggers MCTS at priority ≥5)

**Target (v3.0):**
- Add message types for MCTS statistics export
- Coordination event tracking (combined actions)

---

## Current Status (v2.0 → v3.0 Transition)

### ✅ v2.0 Implemented & Validated
- Command Pipeline (Perception → MCTS → Commands → StateTree)
- MCTS tree search (~34ms, event-driven)
- StateTree execution (all tasks, evaluators, conditions)
- Combat system (Health, Weapon, rewards)
- Perception system (enemy detection, auto-reporting)
- Observations (71+40 features)
- EQS cover system
- Simulation Manager GameMode

### 🔄 v3.0 In Progress (See REFACTORING_PLAN.md)
**Sprint 1-2 (Weeks 1-4): Value Network + World Model**
- [ ] Implement `TeamValueNetwork.h/cpp`
- [ ] Implement `WorldModel.h/cpp`
- [ ] Modify `MCTS.cpp:SimulateNode()` to use both
- [ ] Create training scripts

**Sprint 3-4 (Weeks 5-8): Coupled Training**
- [ ] MCTS → RL: Curriculum Manager
- [ ] RL → MCTS: Policy priors
- [ ] Dual-head `HybridPolicyNetwork`

**Sprint 5-6 (Weeks 9-12): Rewards + Planning**
- [ ] Unified hierarchical rewards
- [ ] UCB action sampling (replace random 10/14,641)
- [ ] Continuous planning (1-2s intervals)
- [ ] Confidence estimates in commands

**Sprint 7 (Weeks 13-14): Self-Play Pipeline**
- [ ] `self_play_collector.py`
- [ ] End-to-end training loop
- [ ] 1000+ self-play games
- [ ] Evaluate vs v2.0 baseline

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

### Architecture Rules (v3.0)
- **Followers NEVER run MCTS** (only leader)
- **MCTS runs continuously** (1-2s intervals, not just events)
- **Value network guides MCTS** (replaces heuristics)
- **World model enables simulation** (predict 5-10 steps)
- **RL provides MCTS priors** (dual-head network)
- **MCTS guides RL curriculum** (prioritized replay on uncertainty)
- **Rewards are aligned** (individual + coordination + strategic)

### Performance Targets (v3.0)
- Team Leader MCTS: 30-50ms (improved with value network pruning)
- RL Inference: 1-3ms (optimized with priors)
- World Model Prediction: 5-10ms (5 steps lookahead)
- StateTree Tick: <0.5ms per agent
- Total Frame Budget: 10-20ms for 4-agent team

---

## File Structure (v3.0)

```
Source/GameAI_Project/
├── MCTS/
│   ├── MCTS.h/cpp                    # 🔄 Modified: ValueNetwork + WorldModel integration
│   ├── TeamMCTSNode.h                # 🔄 Modified: Add ActionPriors
│   └── CommandSynergy.h/cpp          # 🆕 NEW: Synergy score computation
├── RL/
│   ├── RLPolicyNetwork.h/cpp         # 🔄 Modified: Add GetActionPriors()
│   ├── TeamValueNetwork.h/cpp        # 🆕 NEW: Team state value estimation
│   ├── HybridPolicyNetwork.h/cpp     # 🆕 NEW: Dual-head (policy + priors)
│   ├── RewardCalculator.h/cpp        # 🆕 NEW: Unified reward system
│   ├── CurriculumManager.h/cpp       # 🆕 NEW: MCTS-guided training
│   └── RLReplayBuffer.h/cpp          # ✅ Existing (minor changes)
├── Simulation/
│   ├── WorldModel.h/cpp              # 🆕 NEW: State transition predictor
│   └── StateTransition.h             # 🆕 NEW: State delta structs
├── StateTree/                         # 🔄 v3.0 Updated (unified execution)
│   ├── Tasks/                        # ExecuteObjective (replaces ExecuteAssault/Defend/Support/Move/Retreat)
│   ├── Evaluators/                   # SyncCommand, UpdateObservation
│   ├── Conditions/                   # CheckCommandType, CheckTacticalAction, IsAlive
│   └── FollowerStateTreeComponent.h/cpp
├── Combat/                            # ✅ No changes (v2.0 complete)
│   ├── HealthComponent.h/cpp
│   └── WeaponComponent.h/cpp
├── Perception/                        # ✅ No changes (v2.0 complete)
│   └── AgentPerceptionComponent.h/cpp
├── EQS/                               # ✅ No changes (v2.0 complete)
│   ├── Generator/                    # CoverPoints
│   ├── Test/                         # CoverQuality
│   └── Context/                      # CoverEnemies
├── Team/
│   ├── TeamLeaderComponent.h/cpp     # 🔄 Modified: Continuous planning, MCTS stats export
│   ├── FollowerAgentComponent.h/cpp  # 🔄 Modified: Confidence-weighted commands, coordination
│   ├── StrategicCommand.h            # 🔄 Modified: Add uncertainty fields
│   └── TeamCommunicationManager.h/cpp # 🔄 Minor: MCTS stats messages
├── Observation/
│   ├── ObservationElement.h/cpp      # 🔄 Modified: Add Clone(), Serialize()
│   └── TeamObservation.h/cpp         # 🔄 Modified: Add ApplyDelta()
├── Core/
│   └── SimulationManagerGameMode.h/cpp # ✅ No changes
├── Scripts/
│   ├── train_tactical_policy.py     # 🔄 Modified: Prioritized replay, priors
│   ├── train_value_network.py       # 🆕 NEW
│   ├── train_world_model.py         # 🆕 NEW
│   ├── train_coupled_system.py      # 🆕 NEW: End-to-end training loop
│   ├── self_play_collector.py       # 🆕 NEW: Automated data collection
│   └── requirements.txt             # 🔄 Modified: Add dependencies
└── Tests/
    ├── TestValueNetwork.cpp          # 🆕 NEW: Unit tests
    ├── TestWorldModel.cpp            # 🆕 NEW
    └── TestMCTSIntegration.cpp       # 🆕 NEW
```

## Success Metrics (v3.0)

**Quantitative:**
1. **Win Rate**: v3.0 agents beat v2.0 baseline ≥70% in 4v4
2. **MCTS Efficiency**: Reach equivalent solution quality in 50% fewer simulations
3. **Coordination**: ≥30% of kills via coordinated actions
4. **Training Speed**: Converge to strong policy in ≤500 self-play games (vs 2000+ random)

**Qualitative:**
1. **Emergent Tactics**: Flanking, suppression, crossfire patterns
2. **Adaptability**: Handle asymmetric scenarios (3v5, varied unit types)
3. **Robustness**: Graceful degradation when ValueNetwork/WorldModel unavailable

---

## References

**Algorithms:**
- AlphaZero (Silver et al., 2018): Self-play + MCTS + value network
- MuZero (Schrittwieser et al., 2020): Learned world model for planning
- OpenAI Five (Berner et al., 2019): Multi-agent RL at scale
- FuN (Vezhnevets et al., 2017): Feudal networks for hierarchy

**Implementation:**
- Unreal NNE: Neural Network Engine for ONNX inference
- PyTorch: Model training framework
- Ray RLlib: Distributed RL (future work)
