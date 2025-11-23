# Next Steps: RL Training Pipeline with Schola + RLlib

## Current Status Summary

**✅ WORKING:**
- MCTS team-level strategic planning (~34ms per decision)
- State Tree task execution (Assault, Defend, Support, Move, Retreat)
- Combat system (movement, firing, damage, rewards)
- Command pipeline from perception to execution
- Agent proximity/formation (SOLVED)
- Experience collection (stores to JSON)
- **Schola Integration (Phase 1)** - Plugin installed, components created

**⚠️ CURRENT ISSUE:**
RL model is NOT actually learning - uses **rule-based heuristics** as fallback.

---

## ✅ Phase 1 Implementation Complete

The following Schola components have been created:

### C++ Components (UE5)
- `Schola/TacticalObserver.h/cpp` - 71-feature observation via FObservationElement::ToFeatureVector()
- `Schola/TacticalActuator.h/cpp` - 16 discrete actions (ETacticalAction enum)
- `Schola/TacticalRewardProvider.h/cpp` - Reward from combat events

### Python Scripts
- `Scripts/sbdapm_env.py` - Gym wrapper for Schola/RLlib
- `Scripts/train_rllib.py` - RLlib PPO training script
- `Scripts/requirements.txt` - Updated with ray[rllib], gymnasium

### Build Configuration
- `GameAI_Project.Build.cs` - Added "Schola" dependency
- `GameAI_Project.uproject` - Schola, NNE, NNERuntimeORT plugins enabled

## Architecture Overview

```
Training: UE5.6 + Schola ←→ gRPC ←→ OpenAI Gym ←→ RLlib ←→ AWS SageMaker
Inference: UE5.6 + NNE + ONNX Runtime (no Python)
```

See `TOTAL_ACHITECTURE_IMPLEMENTATION_PLAN.md` for full architecture diagram.

---

## Phase 1: Schola Plugin Integration (IMMEDIATE)

**Goal:** Establish gRPC communication between UE and Python/RLlib

### Step 1.1: Install Schola Plugin
```bash
cd C:\Users\Foryoucom\Documents\GitHub\4d\SBDAPM
git clone https://github.com/GPUOpen-LibrariesAndSDKs/Schola.git Plugins/Schola
```

### Step 1.2: Enable Plugins in .uproject
```json
"Plugins": [
    { "Name": "Schola", "Enabled": true },
    { "Name": "NNERuntimeORT", "Enabled": true }
]
```

### Step 1.3: Install Python Package
```bash
pip install schola[rllib]
# Or for all frameworks:
pip install schola[all]
```

### Step 1.4: Create Schola Agent Component
Create `ScholaAgentComponent` on follower pawns that:
- **Sensors:** Maps to existing 71-feature observation (`FObservationElement`)
- **Actuators:** Maps to 16 discrete actions (`ETacticalAction`)
- **Rewards:** Exposes reward from `FollowerAgentComponent`

---

## Phase 2: RLlib Training Setup

**Goal:** Train agents using RLlib PPO via Schola's Gym wrapper

### Step 2.1: Create SBDAPM Environment Wrapper
```python
# Scripts/sbdapm_env.py
from schola.envs import UnrealEnv
from gymnasium import spaces
import numpy as np

class SBDAPMEnv(UnrealEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 71 observation features (follower) + 40 (team) = 111 total
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(71,), dtype=np.float32
        )
        # 16 discrete tactical actions
        self.action_space = spaces.Discrete(16)
```

### Step 2.2: Create RLlib Training Script
```python
# Scripts/train_rllib.py
from ray.rllib.algorithms.ppo import PPOConfig
from sbdapm_env import SBDAPMEnv

config = (
    PPOConfig()
    .environment(SBDAPMEnv, env_config={"host": "localhost", "port": 50051})
    .training(lr=3e-4, train_batch_size=4000, sgd_minibatch_size=128)
    .framework("torch")
)

algo = config.build()
for i in range(100):
    result = algo.train()
    print(f"Episode {i}: reward={result['episode_reward_mean']:.2f}")

# Export to ONNX
algo.export_policy_model("tactical_policy", onnx=True)
```

### Step 2.3: Run Local Training
```bash
# Terminal 1: Start UE with Schola server
UnrealEditor.exe GameAI_Project.uproject -game -windowed -resx=800 -resy=600

# Terminal 2: Run training
cd Source/GameAI_Project/Scripts
python train_rllib.py
```

---

## Phase 3: ONNX Inference in UE (Production)

**Goal:** Load trained model for real-time inference without Python

### Step 3.1: Export ONNX from RLlib
```python
# After training
algo.export_policy_model("Content/Models/tactical_policy", onnx=True)
```

### Step 3.2: Load in UE
```cpp
// In FollowerAgentComponent
if (TacticalPolicy)
{
    bool bLoaded = TacticalPolicy->LoadPolicy(TEXT("Content/Models/tactical_policy.onnx"));
    if (bLoaded)
    {
        UE_LOG(LogTemp, Log, TEXT("ONNX model loaded successfully"));
    }
}
```

### Step 3.3: Verify Inference
- Check `bUseONNXModel` is true
- Inference time should be <5ms
- Actions should differ from rule-based fallback

## File Locations

```
SBDAPM/
├── Plugins/
│   └── Schola/                     # AMD Schola plugin (gRPC bridge)
├── Source/GameAI_Project/
│   ├── Scripts/
│   │   ├── sbdapm_env.py           # ✅ Custom Gym environment wrapper
│   │   ├── train_rllib.py          # ✅ RLlib PPO training script
│   │   ├── train_tactical_policy.py # Offline training from JSON
│   │   └── requirements.txt        # ✅ Python dependencies
│   ├── Public/Schola/              # ✅ NEW - Schola integration
│   │   ├── TacticalObserver.h      # 71-feature observation
│   │   ├── TacticalActuator.h      # 16 discrete actions
│   │   └── TacticalRewardProvider.h # Combat reward provider
│   ├── Private/Schola/             # ✅ NEW
│   │   ├── TacticalObserver.cpp
│   │   ├── TacticalActuator.cpp
│   │   └── TacticalRewardProvider.cpp
│   ├── Private/RL/
│   │   ├── RLPolicyNetwork.cpp     # ONNX loading + inference
│   │   └── RLReplayBuffer.cpp      # Experience storage (offline fallback)
│   └── Public/RL/
│       ├── RLPolicyNetwork.h       # Policy interface
│       └── RLTypes.h               # ETacticalAction, FRLExperience
└── Content/Models/
    └── tactical_policy.onnx        # Trained model (after Phase 2)
```

## Testing Checklist

### Phase 1: Schola Integration
- [x] Schola plugin cloned to `Plugins/Schola`
- [x] NNE + NNERuntimeORT plugins enabled
- [x] C++ components created (TacticalObserver, TacticalActuator, TacticalRewardProvider)
- [ ] Project compiles without errors
- [ ] `pip install schola[rllib]` succeeds
- [ ] Schola server starts when UE launches (check logs for gRPC port)

### Phase 2: RLlib Training
- [ ] Python connects to UE via Schola gRPC
- [ ] Observations received (71 features, valid ranges)
- [ ] Actions sent to UE (0-15 → ETacticalAction)
- [ ] Rewards received from combat events
- [ ] `train_rllib.py` runs without errors
- [ ] Episode reward increases over training

### Phase 3: ONNX Inference
- [ ] ONNX model exported from RLlib
- [ ] `LoadPolicy()` returns true in UE
- [ ] `bUseONNXModel` set to true
- [ ] `SelectAction()` uses neural network (not rule-based)
- [ ] Inference time < 5ms per query

### Phase 4: Evaluation
- [ ] Trained model produces different actions than rule-based
- [ ] Average reward improves over untrained model
- [ ] Agents exhibit learned behaviors (flanking, retreating when low health)

## Troubleshooting

**"Schola plugin not found"**
- Verify `Plugins/Schola` directory exists
- Check `Schola.uplugin` is present
- Rebuild project after adding plugin

**"gRPC connection refused"**
- Ensure UE is running with Schola enabled
- Check default port 50051 is not blocked
- Verify firewall allows local connections

**"NNERuntimeORTCpu not available"**
- Enable NNERuntimeORT plugin in Editor
- Check if plugin is listed in .uproject

**"Failed to create NNE model"**
- Ensure ONNX opset version is 11-13
- Export with `torch.onnx.export(..., opset_version=11)`
- Verify model input shape matches (71,)

**Training loss not decreasing**
- Check reward signal is being sent correctly
- Increase `train_batch_size` in RLlib config
- Reduce learning rate (try 1e-4)
- Ensure episodes are terminating (deaths reset env)

## Success Criteria

**Schola Connection:**
```
[Schola] gRPC server started on port 50051
[Schola] Client connected from 127.0.0.1
[Schola] Observation sent: 71 features
[Schola] Action received: 5 (TakeCover)
```

**RLlib Training:**
```
Episode 0: reward=-12.50
Episode 10: reward=5.30
Episode 50: reward=25.80
Episode 100: reward=42.10
Model exported to: tactical_policy.onnx
```

**ONNX Inference:**
```
[RLPolicy] Loading ONNX model from: Content/Models/tactical_policy.onnx
[RLPolicy] Model loaded successfully
[RLPolicy] Using ONNX inference
[RLPolicy] Selected action: TakeCover (prob: 0.42)
```
