# SBDAPM: AlphaZero-Inspired Multi-Agent Combat AI

**Engine:** UE5.6 | **Language:** C++17 | **Platform:** Windows | **Version:** v3.0 ✅

---

## Quick Reference

### Decision Tree
```
Task Type?
├─ Add Feature → Read affected files → Check design patterns → Implement → Test
├─ Fix Bug → Reproduce → Read stack trace → Locate file:line → Fix → Verify
├─ Optimize → Profile first → Identify bottleneck → Apply pattern → Benchmark
└─ Refactor → Read dependencies → Plan backwards from usage → Implement → Validate
```

### Critical Constraints
| Component | Max Latency | Memory | Notes |
|-----------|-------------|--------|-------|
| MCTS | 30-50ms | 2MB | Value network pruning |
| RL Inference | 1-3ms | 500KB | ONNX/NNE optimized |
| World Model | 5-10ms | 1MB | 5-step lookahead |
| StateTree | <0.5ms/agent | 100KB | Per-tick budget |
| **Total (4 agents)** | **10-20ms** | **8MB** | Real-time requirement |

### File Locations (Quick Jump)
| Feature | Path | Key Methods |
|---------|------|-------------|
| MCTS | `AI/MCTS/MCTS.cpp` | `RunMCTS():71`, `SimulateNode():143` |
| Value Network | `RL/TeamValueNetwork.cpp` | `EvaluateState():34` |
| World Model | `Simulation/WorldModel.cpp` | `PredictNextState():56` |
| RL Policy | `RL/RLPolicyNetwork.cpp` | `SelectAction():89`, `GetActionPriors():127` |
| Team Leader | `Team/TeamLeaderComponent.cpp` | `TickComponent():182`, `ProcessMCTSResults():245` |
| Follower | `Team/FollowerAgentComponent.cpp` | `ExecuteCommand():94` |

### Search Terms (When Unfamiliar)
- **MCTS:** "UCB1 algorithm 2002", "PUCT AlphaGo", "UE5 async task graph"
- **Value Networks:** "TD-learning Sutton", "neural network leaf evaluation"
- **World Models:** "MuZero learned dynamics 2020", "state transition prediction"
- **PPO:** "Proximal Policy Optimization Schulman 2017", "PPO clipping ratio"
- **UE5 APIs:** "UE5.6 NNE ONNX runtime", "UStaticMesh performance", "FTimerManager"

---

## Architecture (v3.0 Complete)

### System Flow
```
Team Leader (1 per team, continuous 1-2s planning)
  ├─ MCTS (value network guided, UCB1 selection)
  │   └─ World Model rollouts (5-10 step lookahead)
  ├─ Confidence Estimates (visit count, value variance, policy entropy)
  └─ Strategic Commands → Followers
                           │
Followers (N agents, tactical execution)
  ├─ RL Policy Network (dual-head: action + MCTS priors)
  ├─ StateTree Execution (command-driven states)
  └─ Coordination Tracking → Unified Rewards → Training Loop
```

### Key Innovations (AlphaZero-Inspired)
1. **Value Network** - Learned state evaluation (no heuristics)
2. **World Model** - True Monte Carlo simulation (predict future states)
3. **Coupled Training** - MCTS guides RL curriculum, RL provides MCTS priors
4. **Unified Rewards** - Hierarchical alignment (individual + coordination + strategic)
5. **Continuous Planning** - Proactive 1-2s intervals with uncertainty quantification

---

## Core Components

### 1. Team Leader (`Team/TeamLeaderComponent.cpp`)
**Role:** Strategic planning via continuous MCTS

**v3.0 Features:**
- Continuous MCTS (1-2s intervals, not event-driven)
- Value network leaf evaluation (replaces `CalculateTeamReward()` heuristics)
- World model rollouts (5-10 steps, stochastic sampling)
- UCB1 action selection (no random sampling)
- Exports uncertainty metrics (visit counts, value variance, policy entropy)

**Commands:** Assault, Defend, Move, Support, Retreat (with confidence estimates)

**Performance:** 30-50ms/tick (500-1000 simulations, pruned by value network)

**Files:** `Team/TeamLeaderComponent.h/cpp`, `AI/MCTS/MCTS.h/cpp`, `AI/MCTS/TeamMCTSNode.h`

---

### 2. Value Network (`RL/TeamValueNetwork.cpp`) ✅
**Purpose:** Guide MCTS tree search via learned state evaluation

**Architecture:**
```
Input: FTeamObservation (40 team + N×71 individual features)
  → Embedding (256, ReLU)
  → Trunk (256→256→128, ReLU)
  → Value Head (128→64→1, Tanh)
Output: State value ∈ [-1, 1] (loss → win probability)
```

**Training:** TD-learning on self-play outcomes (see `Scripts/train_value_network.py`)

**Integration:** `MCTS.cpp:SimulateNode():143` - Query network for leaf nodes

**Search Terms:** "value network AlphaZero", "TD temporal difference learning", "UE5 NNE inference"

---

### 3. World Model (`Simulation/WorldModel.cpp`) ✅
**Purpose:** Predict future states for Monte Carlo rollouts

**Architecture:**
```
Input: State (TeamObs) + Actions (commands + tactical)
  → Action Encoder (embeddings)
  → State Encoder (embeddings)
  → Fusion (concat + MLP)
  → Transition Predictor (state deltas)
Output: NextState (predicted TeamObs)
```

**Predictions:** Health deltas, position changes, status effects (stochastic)

**Training:** Supervised MSE on real transitions (S_t, A_t, S_{t+1}) from gameplay logs

**Integration:** `MCTS.cpp:SimulateNode():167` - Rollout 5-10 steps ahead

**Search Terms:** "MuZero world model", "learned dynamics model", "state transition prediction"

---

### 4. RL Policy Network (`RL/RLPolicyNetwork.cpp` + `RL/HybridPolicyNetwork.cpp`) ✅
**Purpose:** Tactical action selection + MCTS priors

**Dual-Head Architecture (v3.0):**
```
Input: Individual observation (71 features)
  → Shared Trunk (128→128→64, ReLU)
  ├─ Policy Head (Softmax) → Action probabilities (immediate execution)
  └─ Prior Head (Logits) → MCTS node initialization (guide tree search)
```

**Actions:** 16 tactical actions (`ETacticalAction` enum)

**Training:** PPO with prioritized replay (MCTS uncertainty-weighted)

**Unified Rewards (v3.0):**
```cpp
// Individual
+10  Kill, +5 Damage, -5 Take Damage, -10 Death

// Coordination (NEW)
+15  Kill during strategic command
+10  Combined action (crossfire, covering)
+5   Formation maintenance
-15  Disobey strategic command

// Strategic (team-level, MCTS only)
+50  Objective captured, +30 Enemy squad wiped, -30 Own squad wiped
```

**Files:** `RL/RLPolicyNetwork.h/cpp`, `RL/HybridPolicyNetwork.h/cpp`, `RL/RewardCalculator.h/cpp`, `RL/CurriculumManager.h/cpp`

**Search Terms:** "PPO algorithm Schulman", "dual-head network architecture", "prioritized experience replay"

---

### 5. Followers (`Team/FollowerAgentComponent.cpp`) ✅
**Role:** Tactical execution with coordination tracking

**v3.0 Features:**
- Confidence-weighted command execution (low confidence → RL override allowed)
- Coordination detection (combined actions with allies → reward bonuses)
- Experience tagging (mark with MCTS uncertainty for prioritized replay)
- Auto-signals events to leader (enemy spotted, ally killed, low health)

**Integration:** RL policy → StateTree execution → Perception/Combat systems

---

### 6. StateTree Execution (`StateTree/FollowerStateTreeComponent.cpp`) ✅
**Purpose:** Unified command execution (replaces separate assault/defend/support/move/retreat states)

**Structure:**
```
Root
├─ ExecuteObjective (unified task, command-driven)
│   ├─ Evaluators: SyncCommand, UpdateObservation
│   └─ Conditions: CheckCommandType, CheckTacticalAction, IsAlive
```

**Files:**
- `StateTree/Tasks/STTask_ExecuteObjective.h/cpp`
- `StateTree/Evaluators/STEvaluator_SyncCommand.h/cpp`
- `StateTree/Conditions/STCondition_CheckCommandType.h/cpp`
- `StateTree/Conditions/STCondition_CheckTacticalAction.h/cpp`

---

### 7. Observations (`Observation/TeamObservation.cpp`) ✅
**Individual:** 71 features (health, position, velocity, ammo, enemy distances, combat state)

**Team:** 40 features (avg health, formation metrics, command type, confidence)

**v3.0 Methods:**
- `ApplyDelta(FStateTransition)` - Apply world model predictions
- `Clone()` - Deep copy for simulation
- `Serialize()` - Export for training data

---

### 8. Combat System (`Combat/HealthComponent.cpp`, `Combat/WeaponComponent.cpp`) ✅
**No changes from v2.0** - Damage calculation, health management, death events

---

### 9. EQS Cover System (`EQS/*`) ✅
**No changes from v2.0** - Grid/tag-based cover generation, multi-factor scoring

---

### 10. Simulation Manager (`Core/SimulationManagerGameMode.cpp`) ✅
**No changes from v2.0** - Team registration, enemy relationship tracking, actor-team mapping

---

## Design Patterns & Principles

### SOLID Adherence
| Principle | Implementation |
|-----------|----------------|
| **Single Responsibility** | Leader (strategy), Follower (tactics), StateTree (execution) separate |
| **Open/Closed** | `ETacticalAction` enum extensible, `FStrategicCommand` struct modifiable |
| **Liskov Substitution** | `UActorComponent` base for Leader/Follower |
| **Interface Segregation** | `IObservable`, `ICommandReceiver` interfaces |
| **Dependency Inversion** | MCTS depends on `IValueNetwork`, `IWorldModel` abstractions |

### Key Patterns
- **Strategy Pattern:** `RLPolicyNetwork` (action selection strategy)
- **Observer Pattern:** Team communication (leader → followers, event signals)
- **State Pattern:** StateTree states (assault, defend, move, support, retreat)
- **Facade Pattern:** `TeamLeaderComponent` (MCTS + ValueNet + WorldModel facade)
- **Template Method:** `MCTS::SimulateNode()` (expansion, simulation, backprop steps)

---

## Work Instructions (Token Efficiency CRITICAL)

### DO
1. **Read first** - Always read affected files before modifying
2. **Use file:line** - Reference code locations precisely (e.g., `MCTS.cpp:143`)
3. **Implement directly** - Code > planning documents
4. **Batch operations** - Read multiple files in parallel, edit sequentially
5. **Search when uncertain** - Use terms from "Search Terms" section above
6. **Profile before optimizing** - Measure, don't guess

### DON'T
1. **NO verbose reports** - Code-focused only, minimal prose
2. **NO redundant updates** - Don't repeat CLAUDE.md content back
3. **NO long explanations** - Brief comments for complex logic only
4. **NO premature abstraction** - YAGNI principle
5. **NO guessing APIs** - Search "UE5.6 [API name]" if unfamiliar

### Code Style
```cpp
// Good (concise, self-documenting)
float Value = ValueNetwork->Evaluate(State);
if (Value > BestValue) { BestAction = Action; BestValue = Value; }

// Bad (over-commented, verbose)
// Evaluate the current state using the value network to get a score
float StateEvaluationScore = ValueNetworkComponent->EvaluateCurrentState(CurrentState);
// Check if this is better than our best so far
if (StateEvaluationScore > BestValueSoFar) {
    // Update best action and value
    BestActionFound = CurrentAction;
    BestValueSoFar = StateEvaluationScore;
}
```

---

## Architecture Rules (Invariants)

1. **ONLY Leaders run MCTS** (followers NEVER touch MCTS)
2. **MCTS runs continuously** (1-2s timer, not event-driven)
3. **Value network replaces heuristics** (no `CalculateTeamReward()` manual scoring)
4. **World model enables simulation** (predict 5-10 steps, no static evaluation)
5. **RL provides MCTS priors** (`HybridPolicyNetwork::GetActionPriors()`)
6. **MCTS guides RL curriculum** (`CurriculumManager` prioritizes high-uncertainty experiences)
7. **Rewards are hierarchical** (individual + coordination + strategic, aligned across levels)
8. **Commands include confidence** (visit count, value variance, policy entropy fields)

---

## File Structure (v3.0)

```
Source/GameAI_Project/
├── AI/MCTS/
│   ├── MCTS.h/cpp                        # ✅ ValueNet + WorldModel integration
│   ├── TeamMCTSNode.h                    # ✅ ActionPriors added
│   └── CommandSynergy.h/cpp              # ✅ Synergy scoring
├── RL/
│   ├── RLPolicyNetwork.h/cpp             # ✅ GetActionPriors() method
│   ├── TeamValueNetwork.h/cpp            # ✅ NEW v3.0
│   ├── HybridPolicyNetwork.h/cpp         # ✅ NEW v3.0 (dual-head)
│   ├── RewardCalculator.h/cpp            # ✅ NEW v3.0 (unified rewards)
│   ├── CurriculumManager.h/cpp           # ✅ NEW v3.0 (MCTS-guided)
│   └── RLReplayBuffer.h/cpp              # ✅ Prioritized replay
├── Simulation/
│   ├── WorldModel.h/cpp                  # ✅ NEW v3.0
│   └── StateTransition.h                 # ✅ NEW v3.0
├── StateTree/
│   ├── Tasks/STTask_ExecuteObjective.*   # ✅ Unified execution
│   ├── Evaluators/STEvaluator_SyncCommand.*
│   ├── Evaluators/STEvaluator_UpdateObservation.*
│   ├── Conditions/STCondition_CheckCommandType.*
│   ├── Conditions/STCondition_CheckTacticalAction.*
│   ├── Conditions/STCondition_IsAlive.*
│   └── FollowerStateTreeComponent.h/cpp  # ✅ v3.0 updated
├── Team/
│   ├── TeamLeaderComponent.h/cpp         # ✅ Continuous planning
│   ├── FollowerAgentComponent.h/cpp      # ✅ Confidence-weighted execution
│   ├── StrategicCommand.h                # ✅ Confidence fields added
│   └── TeamCommunicationManager.h/cpp    # ✅ MCTS stats messaging
├── Observation/
│   ├── ObservationElement.h/cpp          # ✅ Clone(), Serialize()
│   └── TeamObservation.h/cpp             # ✅ ApplyDelta()
├── Combat/                                # ✅ No changes (v2.0)
├── Perception/                            # ✅ No changes (v2.0)
├── EQS/                                   # ✅ No changes (v2.0)
├── Core/                                  # ✅ No changes (v2.0)
├── Scripts/
│   ├── train_tactical_policy.py          # ✅ Prioritized replay
│   ├── train_value_network.py            # ✅ NEW v3.0
│   ├── train_world_model.py              # ✅ NEW v3.0
│   ├── train_coupled_system.py           # ✅ NEW v3.0
│   ├── self_play_collector.py            # ✅ NEW v3.0
│   └── requirements.txt                  # ✅ Updated dependencies
└── Tests/
    ├── TestValueNetwork.cpp              # ✅ NEW v3.0
    ├── TestWorldModel.cpp                # ✅ NEW v3.0
    └── TestMCTSIntegration.cpp           # ✅ NEW v3.0
```

---

## Success Metrics (v3.0)

### Quantitative
| Metric | Target | Baseline (v2.0) |
|--------|--------|-----------------|
| Win Rate (4v4) | ≥70% vs v2.0 | - |
| MCTS Efficiency | 50% fewer sims for same quality | 500-1000 sims |
| Coordination Rate | ≥30% kills coordinated | ~5% |
| Training Convergence | ≤500 self-play games | ~2000 random games |

### Qualitative
1. **Emergent Tactics:** Flanking, suppression, crossfire without explicit programming
2. **Adaptability:** Handle 3v5, varied unit types gracefully
3. **Robustness:** Fallback to rule-based if ValueNet/WorldModel unavailable

---

## Training Pipeline (v3.0)

### Self-Play Loop
```
1. Collect Data (self_play_collector.py)
   ├─ Run 1000+ games (v3.0 vs v3.0)
   ├─ Log transitions: (S_t, A_t, S_{t+1}, R_t, done)
   └─ Export MCTS statistics (visit counts, values, priors)

2. Train Models (train_coupled_system.py)
   ├─ Value Network (TD-learning on outcomes)
   ├─ World Model (MSE on state transitions)
   └─ RL Policy (PPO with MCTS curriculum)

3. Evaluate
   ├─ v3.0 (new) vs v2.0 (baseline)
   ├─ Win rate, coordination metrics
   └─ If improvement > 5%, replace model; else iterate
```

### Dependencies (requirements.txt)
```
torch==2.0.1
onnx==1.14.0
ray[rllib]==2.6.0  # Future: distributed training
numpy==1.24.3
tensorboard==2.13.0
```

---

## References (Search These First When Uncertain)

### Papers
- **AlphaZero:** "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (Silver et al., 2018)
- **MuZero:** "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (Schrittwieser et al., 2020)
- **PPO:** "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- **UCB1:** "Finite-time Analysis of the Multiarmed Bandit Problem" (Auer et al., 2002)
- **OpenAI Five:** "Dota 2 with Large Scale Deep Reinforcement Learning" (Berner et al., 2019)

### UE5 Documentation
- **NNE ONNX Runtime:** Search "UE5.6 Neural Network Engine ONNX"
- **StateTree:** Search "UE5 StateTree plugin documentation"
- **EQS:** Search "UE5 Environment Query System cover generation"
- **Async Tasks:** Search "UE5 FAsyncTask TAsyncTask performance"

### Algorithms
- **MCTS:** Search "UCB1 tree search", "PUCT AlphaGo upper confidence bound"
- **Value Networks:** Search "neural network state evaluation reinforcement learning"
- **World Models:** Search "learned dynamics model MuZero", "state transition prediction neural network"
- **PPO:** Search "clipping ratio PPO", "advantage estimation GAE"

---

## Common Debugging (Quick Fixes)

| Issue | Likely Cause | Fix Location |
|-------|--------------|--------------|
| MCTS timeout | Value network too slow | `TeamValueNetwork.cpp:34` - Check ONNX model size |
| RL always same action | Priors too strong | `HybridPolicyNetwork.cpp:127` - Reduce prior weight |
| World model drift | Prediction error accumulation | `WorldModel.cpp:56` - Limit rollout depth to 5 steps |
| Followers ignore commands | Confidence too low | `FollowerAgentComponent.cpp:94` - Lower threshold |
| Memory spike | MCTS tree not pruned | `MCTS.cpp:71` - Enable early pruning |

---

## Next Steps (Post v3.0)

### Future Enhancements (v4.0 Candidates)
1. **Distributed Training:** Ray RLlib for multi-GPU self-play
2. **Opponent Modeling:** Predict enemy strategy (Theory of Mind)
3. **Hierarchical MCTS:** Multi-level planning (squad → team → faction)
4. **Transfer Learning:** Pretrain on simpler scenarios, fine-tune on complex
5. **Explainability:** Visualize MCTS trees, value heatmaps in-editor

### Performance Optimizations
- **MCTS Parallelization:** Multi-threaded tree search (currently single-thread)
- **Model Quantization:** INT8 inference (reduce latency 2-3x)
- **Sparse Observations:** Only update changed features (reduce network input size)
- **Compiled Models:** UE5 NNE compiled runtime (vs interpreted ONNX)

---

**Last Updated:** v3.0 Refactoring Complete (2025-11-27)
