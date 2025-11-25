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
- [ ] Implement `TeamValueNetwork.h/cpp`
- [ ] Modify `MCTS.cpp:SimulateNode()` to use ValueNetwork
- [ ] Create `train_value_network.py`
- [ ] Collect initial training data (hand-crafted policies)
- [ ] Train baseline value network

**Validation**: Value network predictions correlate with game outcomes

### Sprint 2 (Weeks 3-4): World Model + True Simulation
- [ ] Implement `WorldModel.h/cpp`
- [ ] Add `FStateTransition` structs
- [ ] Log state transitions during gameplay
- [ ] Train transition predictor
- [ ] Integrate into `MCTS.cpp:SimulateNode()`

**Validation**: Predicted states match actual states within 10% error

### Sprint 3 (Weeks 5-6): Coupled Training (MCTS → RL)
- [ ] Implement `CurriculumManager.h/cpp`
- [ ] Export MCTS statistics (visit counts, values)
- [ ] Prioritized replay in `train_tactical_policy.py`
- [ ] Test on high-variance scenarios

**Validation**: RL converges faster with MCTS curriculum vs random sampling

### Sprint 4 (Weeks 7-8): Policy Priors (RL → MCTS)
- [ ] Add `GetActionPriors()` to `RLPolicyNetwork.h`
- [ ] Modify `MCTS.cpp:ExpandNode()` to use priors
- [ ] Train `HybridPolicyNetwork` with dual heads
- [ ] Benchmark MCTS search depth vs vanilla

**Validation**: MCTS reaches better solutions in fewer simulations

### Sprint 5 (Weeks 9-10): Reward Alignment + UCB Sampling
- [ ] Implement `RewardCalculator.h/cpp`
- [ ] Add coordination bonus tracking
- [ ] Replace `GenerateCommandCombinations()` with UCB version
- [ ] Retrain RL policy with aligned rewards

**Validation**: Agents exhibit coordinated behavior (formation, combined fire)

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
│   ├── MCTS.h/cpp                    # 🔄 Modified: ValueNetwork + WorldModel integration
│   ├── TeamMCTSNode.h                # 🔄 Modified: Add ActionPriors
│   └── CommandSynergy.h/cpp          # 🆕 NEW: Synergy score computation
├── RL/
│   ├── RLPolicyNetwork.h/cpp         # 🔄 Modified: Add GetActionPriors()
│   ├── TeamValueNetwork.h/cpp        # 🆕 NEW: Team state value estimation
│   ├── HybridPolicyNetwork.h/cpp     # 🆕 NEW: Dual-head (policy + priors)
│   ├── RewardCalculator.h/cpp        # 🆕 NEW: Unified reward system
│   └── CurriculumManager.h/cpp       # 🆕 NEW: MCTS-guided training
├── Simulation/
│   ├── WorldModel.h/cpp              # 🆕 NEW: State transition predictor
│   └── StateTransition.h             # 🆕 NEW: State delta structs
├── Team/
│   ├── TeamLeaderComponent.h/cpp     # 🔄 Modified: Continuous planning, MCTS stats export
│   ├── FollowerAgentComponent.h/cpp  # 🔄 Modified: Confidence-weighted commands
│   └── StrategicCommand.h            # 🔄 Modified: Add uncertainty fields
├── Scripts/
│   ├── train_value_network.py        # 🆕 NEW
│   ├── train_world_model.py          # 🆕 NEW
│   ├── train_coupled_system.py       # 🆕 NEW
│   ├── self_play_collector.py        # 🆕 NEW
│   └── curriculum_config.json        # 🆕 NEW
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
