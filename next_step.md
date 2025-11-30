# SBDAPM Next Steps

**Date:** 2025-12-01
**Status:** Environment Testing Complete, MCTS Observation Fix Applied

---

## Completed Work

### 1. Environment Testing ✅
- **File:** `Source/GameAI_Project/Scripts/test_env.py`
- **Status:** All tests passing
- **Validated:**
  - Schola environment connection (localhost:50051)
  - Environment creation (SBDAPMScholaEnv)
  - Observation space: 78 dims (71 obs + 7 objective)
  - Action space: 8 dims (atomic actions)
  - reset() and step() functionality
  - Environment close/cleanup

### 2. Fixed MCTS Individual Observations (MCTS.cpp:178) ✅
- **File:** `Source/GameAI_Project/Private/AI/MCTS/MCTS.cpp`
- **Issue:** MCTS was using placeholder team-level observations instead of actual individual follower observations
- **Fix Applied:**
  - Extract observation directly from `FollowerAgentComponent->GetLocalObservation()`
  - Removed placeholder values (AgentHealth, VisibleEnemyCount, bHasCover, NearestCoverDistance)
  - Now uses real-time, agent-specific observations for PPO critic value estimation
- **Impact:**
  - More accurate MCTS leaf value estimation
  - Better strategic planning quality
  - Individual agent state properly reflected in decision-making
- **Changes:**
  - Added `#include "Team/FollowerAgentComponent.h"`
  - Updated `SimulateNode()` to query follower component directly

**Code Change Summary:**
```cpp
// Before (MCTS.cpp:178-184):
FObservationElement FollowerObs;
FollowerObs.AgentHealth = TeamObs.AverageTeamHealth;
FollowerObs.VisibleEnemyCount = TeamObs.TotalVisibleEnemies;
FollowerObs.bHasCover = TeamObs.bHasCoverAdvantage;
FollowerObs.NearestCoverDistance = 500.0f;  // Placeholder

// After (MCTS.cpp:178-185):
UFollowerAgentComponent* FollowerComponent = Follower->FindComponentByClass<UFollowerAgentComponent>();
if (!FollowerComponent)
{
    continue;
}
FObservationElement FollowerObs = FollowerComponent->GetLocalObservation();
```

---

## Verification Steps (TODO)

### 1. Compile Check ⏳
```bash
# Option A: Build in UE5 Editor
# - Open GameAI_Project.uproject in UE5 Editor
# - Build > Build GameAI_Project
# - Check Output Log for compilation errors

# Option B: MSBuild (Windows)
# MSBuild GameAI_Project.sln /p:Configuration=Development /p:Platform=Win64
```

### 2. Runtime Validation (After Compile) ⏳
- Launch UE5 game mode with Schola server
- Enable MCTS logging (`LogTemp` verbosity)
- Verify individual observations are being used (check log output for follower-specific values)
- Compare MCTS value estimates before/after fix

---

## Next Steps (Recommended)

### High Priority

#### 1. Start Real-Time PPO Training 🎯
**Why:** With observations fixed, training will now use accurate individual agent states.

**Steps:**
```bash
# Terminal 1: Start UE5 game mode with Schola server
# (Launch from UE5 Editor)

# Terminal 2: Start RLlib training
cd Source/GameAI_Project/Scripts
python train_rllib.py --iterations 100 --checkpoint-freq 10
```

**Expected Outcomes:**
- PPO actor-critic network trained via real-time gameplay
- Model export: `training_results/YYYYMMDD_HHMMSS/rl_policy_network.onnx`
- Dual heads: actor (8-dim actions) + critic (1-dim value)
- Checkpoints saved every 10 iterations

**Monitoring:**
```bash
# Start TensorBoard (separate terminal)
tensorboard --logdir Source/GameAI_Project/Scripts/training_results/
# View at http://localhost:6006
```

**Metrics to Track:**
- Episode reward mean (should increase over time)
- Episode length mean
- Policy loss, value loss
- Entropy (exploration level)

#### 2. Performance Profiling 📊
**Why:** Ensure real-time constraints are met (5-10ms total per tick for 4 agents).

**Tools:**
- UE5 Insights (Trace analyzer)
- UE5 Stat commands: `stat game`, `stat unit`, `stat ai`

**Targets:**
- MCTS: 30-50ms per tick (500-1000 simulations)
- RL Inference: 1-3ms per action
- StateTree execution: <0.5ms per agent
- **Total (4 agents): 5-10ms budget**

**Profile Points:**
- `MCTS::SimulateNode()` - Value estimation latency
- `RLPolicyNetwork::GetAction()` - Actor head inference
- `RLPolicyNetwork::GetStateValue()` - Critic head inference
- `STTask_ExecuteObjective::Tick()` - StateTree overhead

#### 3. Deploy Trained Model to UE5 🚀
**After training completes:**
```bash
# Copy best model to UE5 Content directory
cp training_results/YYYYMMDD_HHMMSS/best/rl_policy_network.onnx \
   ../../Content/Models/rl_policy_network.onnx

# Restart UE5 or hot-reload model
# RLPolicyNetwork will automatically load updated model
```

**Verify in UE5:**
- Check `LogTemp` for "RL Policy loaded successfully"
- Observe agent behavior changes (should show learned tactics)
- Run evaluation matches (4v4 vs baseline)

---

### Medium Priority

#### 4. Implement Replay Buffer Persistence (Optional)
**File:** `Source/GameAI_Project/Private/RL/RLReplayBuffer.cpp:352`
**TODO:** Implement proper deserialization for replay buffer loading

**Use Case:**
- Save experiences across training sessions
- Offline analysis of agent behavior
- Curriculum learning (replay past difficult scenarios)

**Not critical for v3.1** - RLlib handles experience collection in real-time.

#### 5. Action Statistics for Continuous Actions
**File:** `Source/GameAI_Project/Private/RL/RLReplayBuffer.cpp:462`
**TODO:** Implement action statistics for continuous action dimensions

**Potential Metrics:**
- Average value per action dimension
- Action variance (exploration indicator)
- Action clipping frequency

**Useful for:**
- Diagnosing training issues (e.g., collapsed policy)
- Tuning action space normalization

---

## File Status Summary

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| **Test Environment** | Scripts/test_env.py | ✅ Passing | All checks green |
| **MCTS Observations** | AI/MCTS/MCTS.cpp | ✅ Fixed | Individual obs extraction |
| **RL Training** | Scripts/train_rllib.py | ⏳ Ready | Needs UE5 server running |
| **Replay Buffer** | RL/RLReplayBuffer.cpp | ⚠️ TODO | Deserialization not critical |
| **Action Stats** | RL/RLReplayBuffer.cpp | ⚠️ TODO | Optional enhancement |

---

## Dependencies Check

**Python Environment:**
```bash
# Verify required packages
pip list | grep -E "ray|torch|schola|numpy"

# Expected:
# ray[rllib]     2.6.0
# torch          2.0.1
# schola[rllib]  (with RLlib integration)
# numpy          1.24.3
```

**UE5 Plugins:**
- Schola 1.3.0 ✅ (verified in Plugins/)
- NNE (Neural Network Engine) - UE5.6 built-in

**gRPC Server:**
- Schola server must run on localhost:50051
- Automatically started by UE5 game mode with Schola plugin

---

## Training Configuration (train_rllib.py)

**Current Hyperparameters:**
```python
LEARNING_RATE = 3e-4
TRAIN_BATCH_SIZE = 4000
SGD_MINIBATCH_SIZE = 128
NUM_SGD_ITER = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_PARAM = 0.2
ENTROPY_COEFF = 0.01
VF_LOSS_COEFF = 0.5
```

**Network Architecture:**
```
Input: 78 features (71 obs + 7 objective)
  → Shared Trunk: [128, 128, 64] (ReLU)
  ├─ Actor Head: 8 dims (atomic actions)
  └─ Critic Head: 1 dim (state value)
```

**Reward Structure (Unified v3.0):**
```cpp
// Individual
+10  Kill, +5 Damage, -5 Take Damage, -10 Death

// Coordination
+15  Kill during strategic command
+10  Combined action (crossfire, covering)
+5   Formation maintenance
-15  Disobey strategic command

// Strategic (MCTS only)
+50  Objective captured
+30  Enemy squad wiped
-30  Own squad wiped
```

---

## Success Metrics (v3.1 Targets)

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Win Rate (4v4)** | ≥70% vs v2.0 | Run 100 matches after training |
| **MCTS Efficiency** | Faster with PPO critic | Profile before/after |
| **Coordination Rate** | ≥30% kills coordinated | Track combined action rewards |
| **Training Convergence** | Continuous improvement | TensorBoard reward curve |
| **Latency** | 5-10ms total/tick | UE5 Insights profiling |

**Qualitative Goals:**
- Emergent tactics (flanking, suppression, crossfire)
- Adaptability (handles 3v5, varied unit types)
- Real-time learning (adapts to opponent strategies)

---

## Known Issues & Limitations

1. **No Replay Buffer Persistence** (RLReplayBuffer.cpp:352)
   - Not critical - RLlib manages experience collection
   - Only needed for offline analysis

2. **Action Statistics Not Implemented** (RLReplayBuffer.cpp:462)
   - Would help diagnose training issues
   - Can be added if policy collapses

3. **MCTS Parallelization**
   - Currently single-threaded tree search
   - Future v4.0 optimization: multi-threaded simulations

4. **Model Quantization**
   - ONNX model uses FP32
   - INT8 quantization could reduce latency 2-3x

---

## Quick Commands Reference

```bash
# Test environment
python Source/GameAI_Project/Scripts/test_env.py

# Train PPO (requires UE5 running)
python Source/GameAI_Project/Scripts/train_rllib.py --iterations 100

# Monitor training
tensorboard --logdir Source/GameAI_Project/Scripts/training_results/

# Analyze experiments (after training)
python Scripts/analyze_experiments.py --csv Saved/Experiments/*.csv --output Results/

# Build UE5 project (Windows)
# Use UE5 Editor: Build > Build GameAI_Project
```

---

## References

- **CLAUDE.md** - Full architecture documentation (v3.1 Real-Time Training)
- **train_rllib.py** - RLlib PPO training script
- **sbdapm_env.py** - Schola environment wrapper
- **MCTS.cpp** - Monte Carlo Tree Search implementation
- **RLPolicyNetwork.cpp** - Actor-critic network (dual-head PPO)

---

**Ready to proceed with training once UE5 Schola server is running!** 🚀
