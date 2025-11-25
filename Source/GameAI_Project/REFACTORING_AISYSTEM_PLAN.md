# REFACTORING PLAN: AlphaZero-Inspired Real-Time Multi-Agent Combat AI

## Design Goal
Transform hierarchical MCTS+RL system into tightly-coupled architecture inspired by AlphaZero, adapted for **real-time, multi-agent, partial observability** combat.

---

## Core Architecture Changes

### Phase 1: Add Value Network (Guide MCTS Tree Search)
**Goal**: Replace hand-crafted heuristics with learned value function

#### 1.1 Create Team Value Network
**New Files:**
- `RL/TeamValueNetwork.h/cpp` - Neural network for team state evaluation

**Architecture:**
```
Input: FTeamObservation (40 team + N×71 individual features)
  ↓ Embedding Layer (256 neurons, ReLU)
  ↓ Shared Trunk (256→256→128, ReLU)
  ↓ Value Head (128→64→1, Tanh)
Output: Team state value [-1, 1] (loss probability → win probability)
```

**Key Features:**
- NNE + ONNX inference (same as RLPolicyNetwork)
- Trained on MCTS rollout outcomes
- Provides value estimates for leaf nodes
- Replaces `CalculateTeamReward()` heuristics

**Files to Modify:**
- `MCTS.cpp:SimulateNode()` - Call ValueNetwork instead of heuristic
- `MCTS.h` - Add `TObjectPtr<UTeamValueNetwork> ValueNetwork`

---

### Phase 2: Add World Model (Enable True Simulation)
**Goal**: Predict future states for Monte Carlo rollouts

#### 2.1 Create State Transition Predictor
**New Files:**
- `Simulation/WorldModel.h/cpp` - Predicts next state given current state + actions
- `Simulation/StateTransition.h` - Structs for state deltas

**Architecture:**
```
Input: CurrentState (TeamObs) + AllActions (strategic + tactical)
  ↓ Action Encoder (commands → embeddings)
  ↓ State Encoder (observations → embeddings)
  ↓ Fusion Layer (concat + MLP)
  ↓ Transition Predictor (outputs state deltas)
Output: NextState (predicted TeamObs)
```

**Predictions:**
- Health changes (damage model)
- Position changes (movement model)
- Status effects (combat outcomes)
- Stochastic sampling for uncertainty

**Training Data:**
- Real game transitions: (S_t, A_t, S_{t+1})
- Supervised learning (MSE loss on state prediction)

**Files to Modify:**
- `MCTS.cpp:SimulateNode()` - Use WorldModel to predict N steps ahead
- `TeamObservation.h` - Add `ApplyDelta(FStateTransition)` method
- `FollowerAgentComponent.cpp` - Log state transitions for training

---

### Phase 3: Coupled Training (MCTS ↔ RL Synergy)
**Goal**: MCTS guides RL training, RL policies guide MCTS

#### 3.1 MCTS → RL: Strategic Curriculum
**New Files:**
- `RL/CurriculumManager.h/cpp` - Prioritizes training scenarios based on MCTS outcomes

**Mechanism:**
1. MCTS identifies high-variance scenarios (uncertain outcomes)
2. CurriculumManager samples these for RL training
3. RL learns from "hard" situations MCTS struggles with

**Files to Modify:**
- `TeamLeaderComponent.cpp` - Export MCTS search statistics (visit counts, values)
- `RLPolicyNetwork.cpp:StoreExperience()` - Tag experiences with MCTS uncertainty
- `train_tactical_policy.py` - Prioritized experience replay weighted by uncertainty

#### 3.2 RL → MCTS: Policy Priors
**New Files:**
- `RL/HybridPolicyNetwork.h/cpp` - Outputs both action probs + prior logits

**Architecture:**
```
Shared Trunk (128→128→64)
  ↓
  ├─→ Policy Head (Softmax) → Immediate action
  └─→ Prior Head (Logits) → MCTS node initialization
```

**Mechanism:**
1. When MCTS expands node, query RL policy for prior probabilities
2. Initialize child visit counts proportional to priors
3. Focuses MCTS search on promising branches

**Files to Modify:**
- `MCTS.cpp:ExpandNode()` - Query RL policy for priors
- `TeamMCTSNode.h` - Add `TArray<float> ActionPriors`
- `RLPolicyNetwork.h` - Add `GetActionPriors()` method

---

### Phase 4: Reward Alignment
**Goal**: Unify strategic and tactical objectives

#### 4.1 Hierarchical Reward Function
**New Files:**
- `RL/RewardCalculator.h/cpp` - Unified reward computation

**New Reward Structure:**
```cpp
// Individual rewards (RL tactical)
+10  Kill enemy
+5   Deal damage
-5   Take damage
-10  Death

// Team coordination bonuses (NEW)
+15  Kill while executing MCTS strategic command
+10  Coordinate with ally (combined fire, cover)
+5   Follow formation
-15  Disobey strategic command (wrong positioning)

// Strategic rewards (MCTS team-level)
+50  Objective captured
+30  Team wipe enemy squad
-30  Team wipe (own squad)
-20  Objective lost
```

**Reward Propagation:**
- RL: Receives immediate + coordination bonuses
- MCTS: Receives discounted team rewards
- ValueNetwork: Trained on MCTS rollout outcomes (end-of-episode team score)

**Files to Modify:**
- `FollowerAgentComponent.cpp:CalculateReward()` - Add coordination checks
- `TeamLeaderComponent.cpp` - Track strategic objective completion
- `MCTS.cpp:BackpropagateNode()` - Discount factor with team rewards

---

### Phase 5: Improved Action Space Sampling
**Goal**: Replace random 10/14,641 sampling with principled selection

#### 5.1 Progressive Widening + UCB
**New Method in MCTS.cpp:**
```cpp
TArray<TMap<AActor*, FStrategicCommand>> GenerateCommandCombinationsUCB(
    const TSharedPtr<FTeamMCTSNode>& ParentNode,
    const TArray<AActor*>& Followers,
    int32 MaxCombinations
);
```

**Algorithm:**
1. **Initial**: Top-K individual commands per follower (K=3)
2. **Composition**: Greedily combine based on predicted synergy
3. **Expansion**: Add random combinations with probability ε
4. **Progressive**: Expand action space as visit count increases

**Implementation:**
- `MCTS.h:79-83` - Replace `GenerateCommandCombinations()`
- Add `CommandSynergy.h/cpp` - Precompute synergy scores

---

### Phase 6: Continuous Planning + Confidence Estimates
**Goal**: Proactive planning with uncertainty quantification

#### 6.1 Time-Sliced MCTS
**Files to Modify:**
- `TeamLeaderComponent.cpp` - Run MCTS every 1-2 seconds (not just events)
- `MCTS.h` - Add `ContinuousPlanningInterval` config

**Mechanism:**
1. Background thread runs MCTS continuously
2. Commands issued when confidence threshold reached
3. Incremental tree reuse (don't rebuild from scratch)

#### 6.2 Uncertainty Estimates
**New Fields in FStrategicCommand:**
```cpp
struct FStrategicCommand {
    EStrategicCommandType CommandType;
    FVector TargetLocation;
    AActor* TargetActor;

    // NEW: Uncertainty quantification
    float Confidence;        // Visit count / total visits
    float ValueVariance;     // Std dev of child values
    float PolicyEntropy;     // H(π) - decision uncertainty
};
```

**Usage:**
- Low confidence → RL can override with tactical judgment
- High variance → Explore more via simulation
- High entropy → Ambiguous situation, gather info

**Files to Modify:**
- `StrategicCommand.h` - Add uncertainty fields
- `MCTS.cpp:SelectNode()` - Compute statistics
- `FollowerAgentComponent.cpp` - Weight commands by confidence

---

## Training Pipeline Changes

### Phase 7: Self-Play + Curriculum
**New Files:**
- `Scripts/self_play_collector.py` - Automated data collection
- `Scripts/train_value_network.py` - Team value network training
- `Scripts/train_world_model.py` - State transition model training
- `Scripts/train_coupled_system.py` - End-to-end training loop

**Self-Play Loop:**
```python
1. Run N games with current policies (RL tactical + MCTS strategic)
2. Collect:
   - RL experiences: (obs, action, reward, next_obs)
   - MCTS traces: (team_obs, commands, visit_counts, final_outcome)
   - State transitions: (team_obs_t, all_actions_t, team_obs_t+1)
3. Train:
   - ValueNetwork on MCTS outcomes (TD-learning)
   - WorldModel on state transitions (supervised)
   - RLPolicy on RL experiences (PPO) with MCTS priors
4. Export models → UE5 NNE
5. Repeat
```

**Curriculum Stages:**
1. **Stage 1**: 1v1 duels (simple)
2. **Stage 2**: 2v2 team fights (coordination)
3. **Stage 3**: 4v4 with objectives (strategic)
4. **Stage 4**: Asymmetric scenarios (adaptation)

---

## Implementation Order

### Sprint 1 (Weeks 1-2): Value Network Foundation
- [x] Implement `TeamValueNetwork.h/cpp`
- [x] Modify `MCTS.cpp:SimulateNode()` to use ValueNetwork
- [x] Create `train_value_network.py`
- [x] Collect initial training data (hand-crafted policies) - `collect_mcts_data.py`
- [ ] Train baseline value network (awaiting data collection)

**Validation**: Value network predictions correlate with game outcomes

**Status**: ✅ COMPLETE. Implementation done. Training awaits gameplay data collection.

### Sprint 2 (Weeks 3-4): World Model + True Simulation
- [x] Implement `WorldModel.h/cpp`
- [x] Add `FStateTransition` structs - `Simulation/StateTransition.h`
- [x] Log state transitions during gameplay - `FollowerAgentComponent::LogStateTransition()`
- [x] Train transition predictor - `train_world_model.py`
- [x] Integrate into `MCTS.cpp:SimulateNode()` - Multi-step rollout with world model

**Validation**: Predicted states match actual states within 10% error

**Status**: ✅ COMPLETE. World model performs 5-step rollouts in MCTS simulation. Training awaits gameplay data.

### Sprint 3 (Weeks 5-6): Coupled Training (MCTS → RL)
- [x] Implement `CurriculumManager.h/cpp`
- [x] Export MCTS statistics (visit counts, values, uncertainty)
- [x] Add MCTS uncertainty tagging to `RLPolicyNetwork::StoreExperience()`
- [x] Prioritized replay in `train_tactical_policy_v3.py`
- [ ] Test on high-variance scenarios

**Validation**: RL converges faster with MCTS curriculum vs random sampling

**Status**: ✅ COMPLETE. Implementation done. Testing awaits gameplay data collection.

### Sprint 4 (Weeks 7-8): Policy Priors (RL → MCTS)
- [x] Add `GetActionPriors()` to `RLPolicyNetwork.h` - Heuristic-based implementation complete
- [x] Modify `MCTS.cpp:ExpandNode()` to use priors - AlphaZero-style PUCT with prior-guided expansion
- [x] Add `ActionPriors` field to `TeamMCTSNode` - Stores priors parallel to UntriedActions
- [x] Integrate RLPolicyNetwork into MCTS - Computes priors for objective assignments
- [x] Implement `HybridPolicyNetwork.h/cpp` stub - Dual-head architecture ready for training
- [ ] Train `HybridPolicyNetwork` with dual heads - Awaits training pipeline
- [ ] Benchmark MCTS search depth vs vanilla - Awaits gameplay testing

**Validation**: MCTS reaches better solutions in fewer simulations

**Status**: ✅ IMPLEMENTATION COMPLETE. Core prior-guided MCTS implemented. Training and benchmarking deferred to gameplay phase.

### Sprint 5 (Weeks 9-10): Reward Alignment + UCB Sampling
- [x] Implement `RewardCalculator.h/cpp` - Hierarchical reward system with individual, coordination, and strategic rewards
- [x] Add coordination bonus tracking - Combined fire, formation, objective adherence tracking
- [x] Replace `GenerateCommandCombinations()` with UCB version - Greedy selection with synergy bonuses and epsilon-greedy exploration
- [ ] Retrain RL policy with aligned rewards - Awaits gameplay data collection

**Validation**: Agents exhibit coordinated behavior (formation, combined fire)

**Status**: ✅ IMPLEMENTATION COMPLETE. RewardCalculator tracks individual (+10 kill, +5 damage, -5 take damage, -10 death), coordination (+15 strategic kill, +10 combined fire, +5 formation, -15 disobey), and strategic rewards (+50 objective complete, +30 enemy wipe, -30 own wipe). MCTS uses UCB-based action sampling with top-3 objectives per follower, synergy bonuses, and 20% exploration. Training awaits gameplay testing.

### Sprint 6 (Weeks 11-12): Continuous Planning + Uncertainty
- [ ] Convert event-driven → time-sliced MCTS
- [ ] Add confidence fields to `FStrategicCommand`
- [ ] Implement confidence-weighted command execution
- [ ] Performance profiling (stay under 10ms/frame)

**Validation**: Proactive planning, smooth command transitions

### Sprint 7 (Weeks 13-14): Self-Play Pipeline
- [ ] Implement `self_play_collector.py`
- [ ] Integrate all training scripts into loop
- [ ] Run 1000+ self-play games
- [ ] Evaluate vs baseline (rule-based heuristics)

**Validation**: Self-play agents outperform hand-crafted policies

---

## File Structure After Refactoring

```
Source/GameAI_Project/
├── MCTS/
│   ├── MCTS.h/cpp                    # ✅ MODIFIED: ValueNetwork + WorldModel + RLPolicy priors (Sprint 1-4 complete)
│   ├── TeamMCTSNode.h/cpp            # ✅ MODIFIED: ActionPriors + PUCT calculation (Sprint 4)
│   └── CommandSynergy.h/cpp          # 🆕 NEW: Synergy score computation (Sprint 5)
├── RL/
│   ├── RLPolicyNetwork.h/cpp         # ✅ MODIFIED: GetObjectivePriors() heuristic-based (Sprint 3-4)
│   ├── TeamValueNetwork.h/cpp        # ✅ IMPLEMENTED: Team state value estimation (Sprint 1)
│   ├── HybridPolicyNetwork.h/cpp     # ✅ IMPLEMENTED: Dual-head stub (Sprint 4, training pending)
│   ├── RewardCalculator.h/cpp        # 🆕 NEW: Unified reward system (Sprint 5)
│   ├── CurriculumManager.h/cpp       # ✅ IMPLEMENTED: MCTS-guided training (Sprint 3)
│   └── RLTypes.h                     # ✅ MODIFIED: Added MCTS uncertainty fields (Sprint 3)
├── Simulation/
│   ├── WorldModel.h/cpp              # ✅ IMPLEMENTED: State transition predictor (Sprint 2)
│   └── StateTransition.h             # ✅ IMPLEMENTED: State delta structs (Sprint 2)
├── Team/
│   ├── TeamLeaderComponent.h/cpp     # ✅ MODIFIED: CurriculumManager integration, MCTS stats export (Sprint 3)
│   ├── FollowerAgentComponent.h/cpp  # ✅ MODIFIED: State transition logging (Sprint 2)
│   └── StrategicCommand.h            # 🔄 Modified: Add uncertainty fields (Sprint 6)
├── Observation/
│   └── TeamObservation.h/cpp         # ✅ MODIFIED: ApplyDelta(), Clone(), Flatten(), Serialize() (Sprint 2)
├── Scripts/
│   ├── train_value_network.py        # ✅ IMPLEMENTED: Value network training (Sprint 1)
│   ├── train_world_model.py          # ✅ IMPLEMENTED: World model training (Sprint 2)
│   ├── train_tactical_policy_v3.py   # ✅ MODIFIED: Prioritized experience replay (Sprint 3)
│   ├── collect_mcts_data.py          # ✅ IMPLEMENTED: Data collection for value network (Sprint 1)
│   ├── train_coupled_system.py       # 🆕 NEW: End-to-end training loop (Sprint 7)
│   ├── self_play_collector.py        # 🆕 NEW: Self-play data collection (Sprint 7)
│   └── curriculum_config.json        # 🆕 NEW: Curriculum configuration (Sprint 3)
└── Tests/
    ├── TestValueNetwork.cpp          # 🆕 NEW: Unit tests
    ├── TestWorldModel.cpp            # 🆕 NEW
    └── TestMCTSIntegration.cpp       # 🆕 NEW
```

---

## Key Architectural Differences: Before vs After

| Aspect | Current (v2.0) | Refactored (v3.0) |
|--------|----------------|-------------------|
| **MCTS Simulation** | Static heuristic evaluation | World model rollouts (5-10 steps) |
| **Value Estimation** | Hand-crafted `CalculateTeamReward()` | Learned `TeamValueNetwork` |
| **Action Sampling** | Random 10/14,641 combinations | UCB + progressive widening |
| **RL ↔ MCTS** | Decoupled, independent | Coupled: Priors + curriculum |
| **Rewards** | Misaligned (individual vs team) | Unified hierarchical rewards |
| **Planning** | Event-driven (reactive) | Continuous (proactive) |
| **Uncertainty** | None | Confidence estimates per command |
| **Training** | Offline RL only | Self-play loop (RL + MCTS + WorldModel) |

---

## Performance Targets (v3.0)

- **MCTS Tree Search**: 30-50ms (improved with value network pruning)
- **RL Inference**: 1-3ms (same, optimized with priors)
- **World Model Prediction**: 5-10ms (5 steps lookahead)
- **Total Frame Budget**: 10-20ms (stay within target)
- **Training Time**: 24-48 hours (1000 self-play games on GPU cluster)

---

## Risk Mitigation

**Risk 1: World Model Inaccuracy**
- Mitigation: Ensemble models (3 predictors, avg predictions)
- Fallback: Blend learned + heuristic predictions (α=0.7 learned, 0.3 heuristic)

**Risk 2: Value Network Overfitting**
- Mitigation: Heavy regularization (dropout 0.3, L2 weight decay)
- Validation: Hold-out test scenarios (never seen in training)

**Risk 3: Training Instability**
- Mitigation: Curriculum (start simple, increase complexity)
- Monitoring: TensorBoard logging (loss, reward curves, policy entropy)

**Risk 4: Real-Time Performance**
- Mitigation: Model quantization (INT8), GPU inference (NNE + CUDA)
- Profiling: Unreal Insights, per-frame breakdown

---

## Success Metrics

**Quantitative:**
1. **Win Rate**: v3.0 agents beat v2.0 baseline ≥70% in 4v4
2. **MCTS Efficiency**: Reach equivalent solution quality in 50% fewer simulations
3. **Coordination**: ≥30% of kills via coordinated actions (combined fire)
4. **Training Speed**: Converge to strong policy in ≤500 self-play games (vs 2000+ random)

**Qualitative:**
1. **Emergent Tactics**: Flanking, suppression, crossfire patterns
2. **Adaptability**: Handle asymmetric scenarios (3v5, varied unit types)
3. **Robustness**: Graceful degradation when ValueNetwork/WorldModel unavailable

---

## Long-Term Extensions (Post-v3.0)

1. **Multi-Team Self-Play**: Red vs Blue vs Green (FFA dynamics)
2. **Meta-Learning**: Adapt to opponent strategies online (MAML)
3. **Explainability**: Visualize MCTS search tree in-editor
4. **Human-AI Teaming**: Mixed human + AI squads
5. **Procedural Scenario Generation**: Auto-create training maps

---

## Implementation Progress Summary

### ✅ Completed Sprints

**Sprint 1 (Weeks 1-2): Value Network Foundation**
- ✅ `TeamValueNetwork.h/cpp` implemented
- ✅ MCTS integration via `SimulateNode()`
- ✅ Training script: `train_value_network.py`
- ✅ Data collection: `collect_mcts_data.py`
- **Status**: Ready for training (awaits gameplay data)

**Sprint 2 (Weeks 3-4): World Model + True Simulation**
- ✅ `WorldModel.h/cpp` implemented
- ✅ `StateTransition.h` structs defined
- ✅ Multi-step rollouts in MCTS (5 steps)
- ✅ State transition logging in `FollowerAgentComponent`
- ✅ Training script: `train_world_model.py`
- ✅ `TeamObservation` extended: `ApplyDelta()`, `Clone()`, `Flatten()`, `Serialize()`
- **Status**: Ready for training (awaits gameplay data)

**Sprint 3 (Weeks 5-6): Coupled Training (MCTS → RL)**
- ✅ `CurriculumManager.h/cpp` implemented
- ✅ MCTS statistics export: `GetMCTSStatistics()`, `GetRootVisitCount()`
- ✅ `TeamLeaderComponent` records scenarios with uncertainty metrics
- ✅ `RLTypes.h` extended with MCTS uncertainty fields
- ✅ `RLPolicyNetwork::StoreExperienceWithUncertainty()` added
- ✅ `train_tactical_policy_v3.py` updated with `PrioritizedSampler`
- ✅ Prioritized experience replay (alpha=0.6, beta=0.4)
- **Status**: Ready for testing (awaits gameplay data collection)

**Sprint 4 (Weeks 7-8): Policy Priors (RL → MCTS)**
- ✅ `GetActionPriors()` in `RLPolicyNetwork` - Heuristic-based context-aware priors
- ✅ `TeamMCTSNode.h` - ActionPriors field + AlphaZero PUCT calculation
- ✅ MCTS prior initialization - Computes priors for objective assignments
- ✅ `HybridPolicyNetwork.h/cpp` - Dual-head architecture stub
- ✅ Prior-guided expansion - Greedy selection based on priors
- **Status**: Implementation complete (training & benchmarking awaits gameplay)

### 🔄 Remaining Sprints

**Sprint 5 (Weeks 9-10): Reward Alignment + UCB Sampling**
- [ ] `RewardCalculator.h/cpp` (unified hierarchical rewards)
- [ ] Coordination bonus tracking
- [ ] UCB action sampling (replace random combinations)

**Sprint 6 (Weeks 11-12): Continuous Planning + Uncertainty**
- [ ] Time-sliced MCTS (1-2s intervals)
- [ ] Confidence fields in `FStrategicCommand`
- [ ] Confidence-weighted command execution

**Sprint 7 (Weeks 13-14): Self-Play Pipeline**
- [ ] `self_play_collector.py`
- [ ] `train_coupled_system.py` (end-to-end loop)
- [ ] 1000+ self-play games
- [ ] Evaluate vs baseline

### Key Achievements (Sprints 1-4)

**Architecture:**
- Value network replaces hand-crafted heuristics in MCTS leaf evaluation
- World model enables true Monte Carlo simulation (5-step lookahead)
- MCTS identifies hard scenarios → RL focuses training on them
- **NEW (Sprint 4)**: RL policy provides priors to guide MCTS tree search (AlphaZero-style)
- **NEW (Sprint 4)**: Prior-guided expansion focuses MCTS on promising branches

**Data Flow:**
```
Gameplay → MCTS (uncertainty metrics) → CurriculumManager → Tagged Experiences
             ↓                                                ↓
      RL Priors (guide tree search)              Prioritized Replay → RL Training
```

**Training Pipeline:**
1. Run gameplay with MCTS-guided agents
2. Export experiences with MCTS uncertainty tags
3. Train RL policy: `python train_tactical_policy_v3.py --use-prioritization`
4. Train value network: `python train_value_network.py`
5. Train world model: `python train_world_model.py`
6. Load trained models back into Unreal (ONNX → NNE)

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
