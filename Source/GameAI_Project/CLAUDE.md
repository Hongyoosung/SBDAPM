# SBDAPM: State-Based Dynamic Action Planning Model - Technical Documentation

## Project Overview

**SBDAPM** (State-Based Dynamic Action Planning Model) is an advanced AI framework for Unreal Engine 5.6 that integrates three complementary techniques for intelligent agent behavior:

1. **Finite State Machines (FSM)** - Structured state management
2. **Monte Carlo Tree Search (MCTS)** - Probabilistic action planning
3. **Reinforcement Learning (RL)** - Reward-based policy optimization

**Engine:** Unreal Engine 5.6
**Language:** C++17
**Platform:** Windows (DirectX 12)
**Total Source Files:** 38 files (19 headers, 19 implementations)

---

## Architecture Overview

### System Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    Unreal Engine Environment                 │
│                   (Game World, Physics, AI)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   StateMachine Component                     │
│              (UActorComponent - Attachable)                  │
│                                                              │
│  ┌────────────┐  ┌─────────────┐  ┌──────────┐            │
│  │ MoveToState│  │ AttackState │  │FleeState │  DeadState │
│  └─────┬──────┘  └──────┬──────┘  └────┬─────┘            │
│        │                 │               │                   │
└────────┼─────────────────┼───────────────┼──────────────────┘
         │                 │               │
         ▼                 ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCTS Decision Engine                      │
│                                                              │
│  1. Selection    → UCT formula selects promising nodes      │
│  2. Expansion    → Create child nodes for actions           │
│  3. Simulation   → Calculate immediate rewards              │
│  4. Backpropagate→ Update tree statistics                   │
│                                                              │
│  Input:  Observation (Health, Distance, Enemies)            │
│  Output: Best Action to Execute                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Action System                           │
│                                                              │
│  Movement:  Forward, Backward, Left, Right                   │
│  Combat:    Skill Attack, Default Attack                     │
│                                                              │
│  Execution: Triggers Blueprint Events for Game Logic        │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Finite State Machine (FSM)

**Files:** `StateMachine.h/cpp`, `State.h/cpp`

#### Purpose
Organizes agent behavior into discrete states, each with specific actions and transitions. The FSM acts as the orchestrator, managing state lifecycle and coordinating with MCTS for decision-making.

#### Key Features
- **Component-Based:** Inherits from `UActorComponent` for flexible attachment to any Actor
- **Blueprint Integration:** Exposes state queries and observation updates to Blueprints
- **State Management:** Maintains 4 states (MoveToState, AttackState, FleeState, DeadState)
- **Observation Tracking:** Monitors environment variables (Health, Distance, Enemies)

#### State Lifecycle
```cpp
// State interface
void EnterState(UStateMachine* StateMachine);        // Initialize state
void UpdateState(UStateMachine* StateMachine, ...);  // Per-frame logic
void ExitState(UStateMachine* StateMachine);         // Cleanup
TArray<UAction*> GetPossibleActions();               // Available actions
```

#### Observations
The FSM tracks a minimal observation space:
- `AgentHealth` (float) - Current health percentage
- `DistanceToDestination` (float) - Distance to goal
- `EnemiesNum` (int32) - Count of nearby enemies

**Limitation:** This is a very limited observation space. Advanced RL typically uses much richer state representations.

---

### 2. Monte Carlo Tree Search (MCTS)

**Files:** `MCTS/MCTS.h/cpp`, `MCTS/MCTSNode.h/cpp`

#### Purpose
Implements tree-based search to explore action sequences and identify optimal paths through probabilistic simulation and exploitation-exploration balance.

#### MCTS Algorithm Phases

##### Phase 1: Selection
```cpp
UMCTSNode* SelectChildNode()
```
Uses Upper Confidence Bound (UCT) formula to select the most promising child node:

```
Score = Exploitation + DynamicExploration × Exploration × ObservationSimilarity

Where:
  Exploitation = TotalReward / VisitCount
  Exploration  = ExplorationParameter × sqrt(ln(Parent.VisitCount) / VisitCount)
  DynamicExploration = f(TreeDepth, AverageReward)
  ObservationSimilarity = exp(-WeightedDistance × 5.0)
```

##### Phase 2: Expansion
```cpp
void Expand(TArray<UAction*> PossibleActions)
```
Creates child nodes for all available actions from the current state. Each child represents a potential action the agent can take.

##### Phase 3: Simulation (Immediate Reward)
```cpp
float CalculateImmediateReward(UMCTSNode* Node)
```
Calculates reward based on observation features:

```cpp
float DistanceReward = 100.0f - Observation.DistanceToDestination;
float HealthReward = Observation.AgentHealth;
float EnemyPenalty = -10.0f * Observation.EnemiesNum;
return DistanceReward + HealthReward + EnemyPenalty;
```

**Issue:** Rewards are hardcoded and may not generalize across different scenarios.

##### Phase 4: Backpropagation
```cpp
void Backpropagate()
```
Propagates rewards up the tree from the selected node to the root, updating:
- Visit counts (incremented)
- Total rewards (accumulated with discount factor)
- Last visit time (for recency bias)

#### Dynamic Parameters

The MCTS implementation includes adaptive parameters:

**Dynamic Exploration Factor:**
```cpp
float CalculateDynamicExplorationParameter()
{
    float TreeDepthFactor = FMath::Max(0.5f, 1.0f - (TreeDepth / 20.0f));
    float RewardFactor = 1.0f; // Adjusted based on average rewards
    return TreeDepthFactor * RewardFactor;
}
```

**Observation Similarity:**
```cpp
float CalculateObservationSimilarity(FObservationElement A, FObservationElement B)
{
    float DistanceWeight = 0.4f;
    float HealthWeight = 0.4f;
    float EnemyWeight = 0.2f;

    float WeightedDistance =
        DistanceWeight * abs(A.Distance - B.Distance) +
        HealthWeight * abs(A.Health - B.Health) +
        EnemyWeight * abs(A.Enemies - B.Enemies);

    return exp(-WeightedDistance * 5.0f);
}
```

#### Configuration Constants
- `ExplorationParameter = 1.41` (√2, standard UCT)
- `DiscountFactor = 0.95` (future reward discounting)
- `MaxTreeDepth = 10` (prevents infinite expansion)

---

### 3. State Implementations

#### MoveToState (Fully Implemented)
**Files:** `States/MoveToState.h/cpp`

**Purpose:** Plans and executes movement towards a destination.

**Actions:**
- `MoveForwardAction`
- `MoveBackwardAction`
- `MoveLeftAction`
- `MoveRightAction`

**Flow:**
1. `EnterState()`: Initialize MCTS, create action list
2. `UpdateState()`: Run MCTS to select best movement action
3. `ExitState()`: Backpropagate results to update tree

#### AttackState (Fully Implemented)
**Files:** `States/AttackState.h/cpp`

**Purpose:** Plans and executes combat actions.

**Actions:**
- `SkillAttackAction` - Special ability attacks
- `DefaultAttackAction` - Basic melee attacks

**Flow:** Similar to MoveToState but with combat-specific actions.

#### FleeState (Stub Implementation)
**Files:** `States/FleeState.h/cpp`

**Status:** Incomplete - no actions defined, no MCTS integration.

**Intended Purpose:** Escape from dangerous situations (low health, overwhelming enemies).

#### DeadState (Stub Implementation)
**Files:** `States/DeadState.h/cpp`

**Status:** Terminal state with no actions. Could be used for death animations, respawn logic, etc.

---

### 4. Action System

**Files:** `Actions/Action.h/cpp`, `Actions/AttackActions/*`, `Actions/MoveToActions/*`

#### Base Action Class
```cpp
class UAction : public UObject
{
public:
    virtual void ExecuteAction(UStateMachine* StateMachine);
};
```

#### Action Types

**Movement Actions:**
All movement actions trigger corresponding Blueprint events:
```cpp
// Example: MoveForwardAction
void UMoveForwardAction::ExecuteAction(UStateMachine* StateMachine)
{
    StateMachine->TriggerBlueprintEvent("MoveF");
}
```

**Attack Actions:**
Currently log execution but delegate actual combat logic to Blueprints:
```cpp
void USkillAttackAction::ExecuteAction(UStateMachine* StateMachine)
{
    UE_LOG(LogTemp, Warning, TEXT("SkillAttack Action Executed"));
    // Blueprint event triggering needed
}
```

**Design Pattern:** Actions serve as lightweight commands that delegate to Blueprint for game-specific implementation. This separates AI decision-making (C++) from game logic (Blueprint).

---

### 5. Observation System

**Files:** `ObservationElement.h/cpp`

#### Structure
```cpp
USTRUCT(BlueprintType)
struct FObservationElement
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite)
    float DistanceToDestination;

    UPROPERTY(BlueprintReadWrite)
    float AgentHealth;

    UPROPERTY(BlueprintReadWrite)
    int32 EnemiesNum;
};
```

**Observation Flow:**
1. Blueprint calls `StateMachine->GetObservation(Health, Distance, Enemies)`
2. StateMachine stores values in member variables
3. MCTS retrieves observations during tree search
4. Observations influence reward calculation and similarity metrics

**Critical Limitation:** Only 3 features. Modern RL systems typically use:
- Vision/sensor data (ray casts, perception)
- Velocity, acceleration
- Animation state
- Inventory/equipment state
- Environmental context (terrain, weather)
- Temporal features (time since last action)

---

## Reinforcement Learning Methodology

### Current Approach

This system implements **model-free, online, value-based reinforcement learning** through MCTS:

#### Learning Mechanism
1. **Value Function:** Implicitly represented by node statistics
   ```
   Q(s, a) = TotalReward / VisitCount
   ```

2. **Policy:** Derived from UCT formula during action selection
   ```
   π(a|s) = argmax[Q(s,a) + c × sqrt(ln(N(s)) / N(s,a))]
   ```

3. **Exploration Strategy:** UCT balances exploration vs exploitation
   - High visit count → exploit (use known good actions)
   - Low visit count → explore (try new actions)

4. **Experience Reuse:**
   - Tree persists across action selections
   - Observation similarity weights historical data
   - Recent visits biased higher (recency factor)

#### Reward Structure

**Immediate Rewards (Non-Terminal):**
```cpp
R = (100 - Distance) + Health + (-10 × Enemies)
```

**Components:**
- **Distance Reward:** Encourages approaching goal
- **Health Reward:** Incentivizes survival
- **Enemy Penalty:** Discourages enemy proximity

**Discount Factor:** γ = 0.95 (balances immediate vs future rewards)

#### Learning Updates

**Per Action Selection:**
1. MCTS explores tree using current observation
2. Best action selected and executed
3. Environment provides new observation
4. Backpropagation updates all visited nodes:
   ```
   Node.TotalReward += Reward × (DiscountFactor ^ Depth)
   Node.VisitCount += 1
   ```

**No Explicit Training Phase:** Learning occurs online during gameplay.

---

## Current Methodology Strengths

### ✅ Advantages

1. **No Training Data Required:** MCTS learns through simulation, no supervised dataset needed

2. **Interpretable Decisions:** Tree structure makes decision path transparent

3. **Handles Sparse Rewards:** Exploration mechanism finds rewards even in difficult scenarios

4. **Dynamic Adaptation:** Adjusts to new situations without retraining

5. **Modular Design:** FSM + MCTS + Actions are loosely coupled and extensible

6. **Unreal Integration:** Component-based design integrates seamlessly with UE architecture

7. **Stochastic Robustness:** MCTS handles uncertainty through probabilistic exploration

---

## Critical Issues & Limitations

### 🔴 Major Issues

#### 1. **LearningAgents Plugin Unused**
**Severity:** High
**File:** `GameAI_Project.uproject:27-29`

The Unreal Engine LearningAgents plugin is enabled but **completely unused**. This plugin provides:
- Imitation learning
- Reinforcement learning with PPO (Proximal Policy Optimization)
- Continuous/discrete action spaces
- Neural network policies
- Distributed training infrastructure

**Current Impact:** Project reinvents RL infrastructure instead of leveraging proven framework.

**Recommendation:** Either integrate LearningAgents or remove the plugin dependency.

---

#### 2. **Poor Code Organization**
**Severity:** High
**Location:** Entire `Source/GameAI_Project/` directory

**Problems:**
- Headers and implementations in same directories
- No Public/Private separation
- No clear API boundary
- Difficult to maintain and extend

**Standard Unreal Structure Should Be:**
```
Source/GameAI_Project/
├── Public/              # API headers (.h)
│   ├── Core/
│   ├── States/
│   ├── Actions/
│   └── MCTS/
└── Private/             # Implementations (.cpp) + internal headers
    ├── Core/
    ├── States/
    ├── Actions/
    └── MCTS/
```

**Impact:**
- Increases build times (all headers exposed)
- Unclear what's intended for external use
- Violates Unreal Engine conventions

---

#### 3. **Incomplete State Implementations**
**Severity:** Medium
**Files:** `States/FleeState.h/cpp`, `States/DeadState.h/cpp`

Both FleeState and DeadState are stub implementations:
- No actions defined
- No MCTS integration
- No meaningful behavior

**Impact:** Agent cannot properly handle retreat scenarios or death, limiting tactical diversity.

---

#### 4. **Hardcoded Parameters**
**Severity:** Medium
**Location:** `MCTS/MCTS.cpp`

Critical parameters are hardcoded:
```cpp
ExplorationParameter = 1.41;    // Line 8
DiscountFactor = 0.95;          // Line 9
MaxTreeDepth = 10;              // Throughout
```

**Problems:**
- Cannot be adjusted without recompilation
- No tuning for different scenarios
- No Blueprint exposure for designers

**Recommendation:** Use `UPROPERTY(EditAnywhere, Category="MCTS")` for runtime editing.

---

#### 5. **Limited Observation Space**
**Severity:** High
**File:** `ObservationElement.h`

Only 3 features: Health, Distance, Enemies.

**Missing Critical Information:**
- Agent velocity/momentum
- Terrain/environmental context
- Recent action history
- Enemy positions (only count, not locations)
- Weapon state/cooldowns
- Animation state

**Impact:** Agent makes decisions with incomplete information, limiting effectiveness.

---

#### 6. **No Neural Network Integration**
**Severity:** High
**Current:** Pure MCTS with handcrafted reward function

**Limitation:** Cannot learn complex patterns or generalize across diverse scenarios. Modern RL uses neural networks for:
- Value function approximation
- Policy representation
- Feature extraction from raw observations

**Example:** AlphaZero uses MCTS + Neural Networks for superhuman performance.

---

#### 7. **No Distributed Learning Infrastructure**
**Severity:** Medium

**Current:** Single-agent, single-process learning.

**Limitations:**
- Slow experience collection
- Cannot leverage multiple CPU cores
- No cloud training support

**Modern Approach:** Distributed frameworks like RLlib, Ray, or AWS SageMaker for:
- Parallel environment simulation
- Distributed experience collection
- Scalable training

---

#### 8. **No Persistence/Checkpointing**
**Severity:** Medium

**Problem:** MCTS tree is **not saved** between sessions. All learned experience is lost when the game closes.

**Impact:** Agent starts from scratch every time, preventing long-term improvement.

**Solution:** Serialize MCTS tree or transition to neural network policy that can be saved/loaded.

---

#### 9. **Synchronous Execution**
**Severity:** Low
**File:** `StateMachine.cpp:TickComponent()`

MCTS runs synchronously in the game tick, potentially causing frame rate drops if tree search is expensive.

**Recommendation:** Run MCTS on background thread, asynchronously updating policy.

---

#### 10. **No Training/Evaluation Separation**
**Severity:** Medium

**Current:** Agent always explores (no "deployment" mode).

**Best Practice:**
- **Training Mode:** High exploration, collect experience
- **Evaluation Mode:** Greedy policy, measure performance
- **Deployment Mode:** Frozen policy, no learning

**Impact:** Cannot objectively measure agent performance improvement.

---

#### 11. **Minimal Unit Testing**
**Severity:** Low

No test coverage for:
- MCTS algorithm correctness
- State transition logic
- Reward calculation
- Observation similarity

**Recommendation:** Implement unit tests using Unreal's Automation Framework.

---

#### 12. **Blueprint Dependency for Actions**
**Severity:** Low
**Files:** All Action classes

Actions trigger Blueprint events but don't implement actual movement/combat logic in C++.

**Pros:** Flexible for designers
**Cons:**
- Debugging split across C++/Blueprint
- Blueprint events may not exist
- Harder to version control

**Recommendation:** Provide C++ implementations with Blueprint override option.

---

## Improvement Opportunities

### 🔵 Proposed Enhancements

#### 1. **Integrate Unreal LearningAgents Plugin**
Replace custom MCTS with LearningAgents framework:
- PPO-based training
- Neural network policies
- Better scalability
- Built-in training tools

#### 2. **Add Distributed Training with RLlib**
Integrate Ray RLlib for distributed reinforcement learning:
```python
# Example RLlib trainer
from ray.rllib.algorithms.ppo import PPOConfig

config = PPOConfig()
    .environment("UnrealEnv")
    .training(
        lr=0.0003,
        train_batch_size=4000,
        num_sgd_iter=30
    )
    .resources(num_gpus=1)
    .rollouts(num_rollout_workers=8)

trainer = config.build()
```

#### 3. **Docker Containerization**
Create Docker images for:
- Training environments
- Evaluation servers
- Cloud deployment

```dockerfile
FROM unrealengine/unreal-engine:5.6

COPY Source/ /app/Source/
RUN UnrealBuildTool GameAI_Project Linux Development
CMD ["./GameAI_Project"]
```

#### 4. **AWS SageMaker Integration**
Enable cloud training:
- Spin up training jobs on-demand
- Use GPU instances (p3/p4)
- Automatic model versioning
- Hyperparameter tuning

#### 5. **Expand Observation Space**
```cpp
struct FObservationElement
{
    // Movement
    FVector Velocity;
    FVector Acceleration;

    // Vision (ray casts)
    TArray<float> RaycastDistances;

    // Combat
    float WeaponCooldown;
    float StaminaPercent;

    // Existing
    float AgentHealth;
    float DistanceToDestination;
    int32 EnemiesNum;

    // Environmental
    ETerrainType TerrainType;
    float TimeOfDay;
};
```

#### 6. **Neural Network Policy**
Replace MCTS with deep RL:
```cpp
class UNeuralNetworkPolicy : public UObject
{
    // Forward pass
    TArray<float> PredictActionProbabilities(FObservationElement Obs);

    // Training
    void UpdateWeights(TArray<FExperience> Batch);
};
```

#### 7. **Experience Replay Buffer**
Store and reuse past experiences:
```cpp
class UReplayBuffer : public UObject
{
    void AddExperience(FObservationElement, UAction*, float Reward);
    TArray<FExperience> SampleBatch(int32 BatchSize);
};
```

#### 8. **Configurable Parameters**
Expose to Blueprint:
```cpp
UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MCTS")
float ExplorationParameter = 1.41f;

UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MCTS")
float DiscountFactor = 0.95f;

UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MCTS")
int32 MaxTreeDepth = 10;
```

#### 9. **Reward Shaping Interface**
Allow designers to customize rewards:
```cpp
UCLASS(Blueprintable)
class URewardFunction : public UObject
{
    UFUNCTION(BlueprintNativeEvent)
    float CalculateReward(FObservationElement Obs, UAction* Action);
};
```

#### 10. **Async MCTS Execution**
```cpp
void UStateMachine::TickComponent(float DeltaTime, ...)
{
    if (!bMCTSRunning)
    {
        AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [this]()
        {
            MCTS->RunSearch();
            bMCTSRunning = false;
        });
        bMCTSRunning = true;
    }
}
```

---

## Build & Deployment

### Prerequisites
- Unreal Engine 5.6
- Visual Studio 2022
- Windows 10/11 SDK
- 16GB+ RAM (for training)

### Building
```bash
# Generate project files
UnrealBuildTool -projectfiles -project="GameAI_Project.uproject"

# Build (Development)
UnrealBuildTool GameAI_Project Win64 Development -project="GameAI_Project.uproject"

# Build (Shipping)
UnrealBuildTool GameAI_Project Win64 Shipping -project="GameAI_Project.uproject"
```

### Running
```bash
# Editor
UnrealEditor.exe "GameAI_Project.uproject"

# Standalone
GameAI_Project.exe
```

---

## Future Roadmap

### Phase 1: Code Refactoring (Weeks 1-2)
- [ ] Reorganize into Public/Private structure
- [ ] Complete FleeState and DeadState implementations
- [ ] Add unit tests for core components
- [ ] Expose parameters to Blueprint

### Phase 2: Enhanced RL (Weeks 3-5)
- [ ] Integrate LearningAgents plugin
- [ ] Implement neural network policy
- [ ] Add experience replay buffer
- [ ] Expand observation space

### Phase 3: Distributed Training (Weeks 6-8)
- [ ] RLlib integration
- [ ] Docker containerization
- [ ] AWS SageMaker training pipeline
- [ ] Model versioning and checkpointing

### Phase 4: Advanced Features (Weeks 9-12)
- [ ] Multi-agent coordination
- [ ] Curriculum learning
- [ ] Inverse reinforcement learning
- [ ] Explainability tools (visualize decision trees)

---

## Contributing

### Code Style
- Follow Unreal Engine coding standards
- Use PascalCase for classes, camelCase for variables
- Prefix Unreal classes with U (UObject), A (AActor), F (struct)
- Comment complex algorithms (especially MCTS)

### Pull Request Process
1. Create feature branch from `main`
2. Implement changes with tests
3. Run `clang-format` on modified files
4. Submit PR with detailed description
5. Ensure CI builds pass

---

## References

### Academic Papers
1. **MCTS:** Browne et al., "A Survey of Monte Carlo Tree Search Methods" (2012)
2. **AlphaZero:** Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (2017)
3. **UCT:** Kocsis & Szepesvári, "Bandit based Monte-Carlo Planning" (2006)

### Frameworks
- [Unreal Learning Agents](https://docs.unrealengine.com/5.6/en-US/learning-agents-in-unreal-engine/)
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)
- [AWS SageMaker RL](https://docs.aws.amazon.com/sagemaker/latest/dg/reinforcement-learning.html)

### Unreal Engine Resources
- [AI Module Documentation](https://docs.unrealengine.com/5.6/en-US/artificial-intelligence-in-unreal-engine/)
- [Component Architecture](https://docs.unrealengine.com/5.6/en-US/components-in-unreal-engine/)
- [Blueprint/C++ Interaction](https://docs.unrealengine.com/5.6/en-US/blueprint-and-c-in-unreal-engine/)

---

## License

This project follows standard Unreal Engine licensing. Consult Epic Games' EULA for distribution rights.

---

## Contact & Support

For questions, issues, or contributions:
- **GitHub Issues:** [Repository Issues Page]
- **Unreal Engine Forums:** [AI & Navigation Section]
- **Discord:** [Project Discord Server]

---

**Last Updated:** 2025-10-26
**Version:** 1.0
**Maintained By:** Claude Code Assistant
