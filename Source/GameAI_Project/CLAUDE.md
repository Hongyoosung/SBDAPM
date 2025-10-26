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

**Hybrid Architecture: Strategic AI + Tactical Execution**

SBDAPM uses a layered architecture separating high-level strategic decision-making from low-level tactical execution:

```
┌─────────────────────────────────────────────────────────────┐
│                    Unreal Engine Environment                 │
│                   (Game World, Physics, AI)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              STRATEGIC LAYER (High-Level Decisions)          │
│                                                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │           StateMachine Component                  │      │
│  │         (Strategic State Management)              │      │
│  │                                                    │      │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐      │      │
│  │  │MoveToState│  │AttackState│  │FleeState │      │      │
│  │  └─────┬────┘  └─────┬─────┘  └────┬─────┘      │      │
│  │        │             │              │             │      │
│  └────────┼─────────────┼──────────────┼─────────────┘      │
│           │             │              │                     │
│           ▼             ▼              ▼                     │
│  ┌─────────────────────────────────────────────────┐       │
│  │         MCTS Decision Engine + RL Policy         │       │
│  │                                                   │       │
│  │  Input:  71-feature Observation Vector           │       │
│  │  Process: MCTS tree search OR Neural Network     │       │
│  │  Output: Strategic Decision (Behavior to execute)│       │
│  │                                                   │       │
│  │  Decisions:                                       │       │
│  │  - Which strategy? (Aggressive/Defensive/Evasive)│       │
│  │  - Which behavior tree subtree to activate?      │       │
│  │  - When to transition states?                    │       │
│  └───────────────────┬───────────────────────────────┘       │
└────────────────────────┼────────────────────────────────────┘
                         │ (Blackboard key: "SelectedStrategy")
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              TACTICAL LAYER (Low-Level Execution)            │
│                                                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │           Unreal Behavior Tree                    │      │
│  │         (Action Execution & Coordination)         │      │
│  │                                                    │      │
│  │  Root Selector                                    │      │
│  │  ├─ Flee Behavior Subtree                         │      │
│  │  │  ├─ Find Cover (EQS)                           │      │
│  │  │  ├─ Sprint to Cover (MoveTo)                   │      │
│  │  │  └─ Evasive Movement (Custom Task)             │      │
│  │  │                                                 │      │
│  │  ├─ Attack Behavior Subtree                       │      │
│  │  │  ├─ Find Firing Position (EQS)                 │      │
│  │  │  ├─ Aim at Target (RotateToFaceBBEntry)        │      │
│  │  │  ├─ Execute Attack (Custom Task)               │      │
│  │  │  └─ Strafe/Dodge (MoveTo)                      │      │
│  │  │                                                 │      │
│  │  └─ MoveTo Behavior Subtree                       │      │
│  │     ├─ Find Path to Destination (MoveTo)          │      │
│  │     ├─ Avoid Obstacles (EQS + MoveTo)             │      │
│  │     └─ Update Progress (Custom Task)              │      │
│  │                                                    │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │         Behavior Tree Tasks & Services           │      │
│  │                                                   │      │
│  │  Tasks:                                           │      │
│  │  - BTTask_ExecuteAttack                           │      │
│  │  - BTTask_UpdateObservation                       │      │
│  │  - BTTask_EvasiveMovement                         │      │
│  │                                                   │      │
│  │  Services:                                        │      │
│  │  - BTService_UpdateThreatAssessment               │      │
│  │  - BTService_SyncObservationToBlackboard          │      │
│  │                                                   │      │
│  │  Decorators:                                      │      │
│  │  - BTDecorator_CheckStrategy (reads Blackboard)   │      │
│  │  - BTDecorator_HealthThreshold                    │      │
│  └───────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

**Strategic Layer (FSM + MCTS/RL):**
- Analyze 71-feature observation space
- Decide high-level strategy (when to attack vs. flee vs. move)
- Select which Behavior Tree subtree to activate
- Learn long-term policies through reinforcement learning
- Update Blackboard with strategic decisions

**Tactical Layer (Behavior Tree + EQS):**
- Execute pathfinding and navigation
- Handle combat mechanics (aiming, shooting, reloading)
- Coordinate animations and abilities
- Query environment (find cover, firing positions)
- React to immediate obstacles and threats

### Key Integration Points

1. **Blackboard Communication:** Strategic layer writes decisions to Blackboard keys
2. **Observation Updates:** BTService periodically updates observation data
3. **Strategy Execution:** BTDecorator checks Blackboard to determine which subtree runs
4. **Reward Feedback:** Behavior Tree task completion feeds rewards back to MCTS/RL

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

### 3. State Implementations (Strategic Layer)

**UPDATED ARCHITECTURE:** States now control high-level strategy and activate Behavior Tree subtrees, rather than directly executing actions.

#### MoveToState (Updated for BT Integration)
**Files:** `States/MoveToState.h/cpp`

**Purpose:** Strategic navigation state - determines *when* and *how* to approach destination.

**MCTS Decisions:**
- Which movement strategy? (Direct, Cautious, Stealth)
- Should we engage enemies encountered en route?
- Should we detour for resources?

**Behavior Tree Integration:**
- Sets Blackboard key: `CurrentStrategy` = "MoveTo"
- Activates MoveTo subtree in Behavior Tree
- BT handles pathfinding, obstacle avoidance, actual movement

**Flow:**
1. `EnterState()`: Initialize MCTS, set Blackboard strategy
2. `UpdateState()`: Run MCTS to select movement strategy, update Blackboard
3. `ExitState()`: Backpropagate results, clear strategy

#### AttackState (Updated for BT Integration)
**Files:** `States/AttackState.h/cpp`

**Purpose:** Strategic combat state - determines *when* and *how* to engage enemies.

**MCTS Decisions:**
- Which combat strategy? (Aggressive, Defensive, Flanking)
- Which target to prioritize?
- When to retreat or reposition?

**Behavior Tree Integration:**
- Sets Blackboard keys: `CurrentStrategy` = "Attack", `TargetEnemy` = selected enemy
- Activates Attack subtree in Behavior Tree
- BT handles aiming, firing, dodging, ability execution

**Flow:** Similar to MoveToState but with combat-specific strategic decisions.

#### FleeState (Requires Implementation)
**Files:** `States/FleeState.h/cpp`

**Status:** **INCOMPLETE** - needs strategic decision logic and BT integration.

**Intended Purpose:** Strategic retreat state - determines optimal escape strategy.

**MCTS Decisions Needed:**
- Which flee strategy? (Sprint to cover, Evasive movement, Call for help)
- Which cover location to target?
- Should we fight back while retreating?

**Behavior Tree Integration Needed:**
- Set Blackboard keys: `CurrentStrategy` = "Flee", `CoverLocation` = target
- Activate Flee subtree in Behavior Tree
- BT handles pathfinding to cover, evasive movement, sprinting

#### DeadState (Terminal State)
**Files:** `States/DeadState.h/cpp`

**Status:** Terminal state with no strategic decisions.

**Behavior Tree Integration:**
- Sets Blackboard: `CurrentStrategy` = "Dead"
- Stops Behavior Tree execution
- Triggers death animation, respawn logic (if applicable)

---

### 4. Behavior Tree System (Tactical Layer)

**NEW ARCHITECTURE COMPONENT:** Unreal Engine Behavior Trees handle all tactical execution.

#### Overview

The Behavior Tree is responsible for:
- **Pathfinding & Navigation:** Using NavMesh and MoveTo tasks
- **Action Execution:** Attack, dodge, interact with objects
- **Environment Queries:** Using EQS to find cover, firing positions, etc.
- **Animation Coordination:** Triggering appropriate animation montages
- **Immediate Reactions:** Responding to damage, obstacles, etc.

#### Key Behavior Tree Components

**Tasks (Custom C++ implementations needed):**
```cpp
// Execute an attack based on current weapon
class UBTTask_ExecuteAttack : public UBTTaskNode
{
    // Reads Blackboard: CurrentWeapon, TargetEnemy
    // Executes: Aim, fire, play animation
    // Returns: Success/Failure
};

// Update observation data for strategic layer
class UBTTask_UpdateObservation : public UBTTaskNode
{
    // Gathers: Enemy positions, health, ammo, etc.
    // Updates: StateMachine->UpdateObservation(...)
};

// Perform evasive movement pattern
class UBTTask_EvasiveMovement : public UBTTaskNode
{
    // Executes: Zigzag movement, dodge roll, etc.
};
```

**Services (Continuous monitoring):**
```cpp
// Update threat level and enemy tracking
class UBTService_UpdateThreatAssessment : public UBTService
{
    // Every tick: Scan for enemies, update danger level
    // Updates Blackboard: NearestEnemy, ThreatLevel
};

// Sync observation data to Blackboard for BT decisions
class UBTService_SyncObservationToBlackboard : public UBTService
{
    // Every tick: Read StateMachine->CurrentObservation
    // Updates Blackboard keys for tactical decisions
};
```

**Decorators (Conditional execution):**
```cpp
// Check if current strategy matches required strategy
class UBTDecorator_CheckStrategy : public UBTDecorator
{
    UPROPERTY(EditAnywhere)
    FString RequiredStrategy; // "Attack", "Flee", "MoveTo"

    // Reads Blackboard: CurrentStrategy
    // Returns: true if matches RequiredStrategy
};
```

#### Behavior Tree Structure

**Blueprint Asset:** `BT_SBDAPM_Agent.uasset`

```
Root (Selector)
├─ [Decorator: IsDead?] DeadBehavior
├─ [Decorator: Strategy == "Flee"] FleeBehavior
│  ├─ [Service: UpdateThreatAssessment] Sequence
│  │  ├─ Task: Find Cover (EQS)
│  │  ├─ Task: MoveTo Cover
│  │  └─ Task: Evasive Movement
├─ [Decorator: Strategy == "Attack"] AttackBehavior
│  ├─ [Service: UpdateTargetTracking] Sequence
│  │  ├─ Task: Find Firing Position (EQS)
│  │  ├─ Task: MoveTo Position
│  │  ├─ Task: Aim at Target
│  │  └─ Task: Execute Attack
└─ [Decorator: Strategy == "MoveTo"] MoveToBehavior
   ├─ [Service: UpdateObservation] Sequence
   │  ├─ Task: Find Path (MoveTo)
   │  └─ Task: Follow Path
```

#### Integration with Strategic Layer

**Blackboard Keys:**
- `CurrentStrategy` (String): Set by FSM states, read by BT decorators
- `TargetEnemy` (Actor): Set by AttackState MCTS, used by attack tasks
- `CoverLocation` (Vector): Set by FleeState MCTS, used by flee tasks
- `Destination` (Vector): Set by MoveToState, used by navigation tasks
- `ObservationData` (Struct): Synced from StateMachine for BT services

**Execution Flow:**
1. Strategic layer (FSM + MCTS) analyzes observation
2. Selects optimal strategy and updates Blackboard
3. BT decorators activate appropriate subtree based on strategy
4. BT tasks execute tactical actions
5. BT services update observation data
6. Strategic layer reads results, updates policy

---

### 5. Observation System

**Files:** `ObservationElement.h/cpp`

**STATUS:** ✅ **UPDATED** - Now provides 71 features (up from original 3)

#### Enhanced Structure (71 Features)

The observation system has been significantly expanded to provide rich environmental and agent state information:

```cpp
USTRUCT(BlueprintType)
struct FObservationElement
{
    GENERATED_BODY()

    // AGENT STATE (12 features)
    FVector Position;           // 3: X, Y, Z
    FVector Velocity;           // 3: VX, VY, VZ
    FRotator Rotation;          // 3: Pitch, Yaw, Roll
    float Health;               // 1: 0-100
    float Stamina;              // 1: 0-100
    float Shield;               // 1: 0-100

    // COMBAT STATE (3 features)
    float WeaponCooldown;       // 1: seconds remaining
    int32 Ammunition;           // 1: bullets/charges
    int32 CurrentWeaponType;    // 1: weapon ID

    // ENVIRONMENT PERCEPTION (32 features)
    TArray<float> RaycastDistances;        // 16: normalized distances
    TArray<ERaycastHitType> RaycastHitTypes; // 16: object types detected

    // ENEMY INFORMATION (16 features)
    int32 VisibleEnemyCount;              // 1: total visible
    TArray<FEnemyObservation> NearbyEnemies; // 5×3=15: closest enemies

    // TACTICAL CONTEXT (5 features)
    bool bHasCover;             // 1: cover available?
    float NearestCoverDistance; // 1: distance to cover
    FVector2D CoverDirection;   // 2: normalized direction
    ETerrainType CurrentTerrain;// 1: terrain type enum

    // TEMPORAL FEATURES (2 features)
    float TimeSinceLastAction;  // 1: seconds elapsed
    int32 LastActionType;       // 1: action ID

    // LEGACY (1 feature - backward compatibility)
    float DistanceToDestination; // 1: distance to goal

    // Utility functions
    TArray<float> ToFeatureVector() const;  // Returns 71 normalized values
    void Reset();
    int32 GetFeatureCount() const { return 71; }
};
```

#### Supporting Structures

**FEnemyObservation:**
```cpp
USTRUCT(BlueprintType)
struct FEnemyObservation
{
    float Distance;        // Distance to enemy
    float Health;          // Enemy's health percentage
    float RelativeAngle;   // Angle from agent's forward (-180 to 180)
};
```

**Enums:**
- `ERaycastHitType`: None, Wall, Enemy, Cover, HealthPack, Weapon, Other
- `ETerrainType`: Flat, Inclined, Rough, Water, Unknown

#### Observation Flow

**Blueprint → Strategic Layer:**
1. BT Service gathers environmental data each tick
2. Calls `StateMachine->UpdateObservation(FObservationElement)`
3. Or uses granular updates: `UpdateAgentState()`, `UpdateCombatState()`, etc.

**Strategic Layer Usage:**
1. FSM states access `StateMachine->CurrentObservation`
2. MCTS uses observations for:
   - Reward calculation
   - Observation similarity (for tree reuse)
   - State evaluation

**Feature Normalization:**
- All features normalized to [0, 1] range via `ToFeatureVector()`
- Ready for neural network input (Phase 2)
- Consistent scaling improves MCTS performance

#### Key Improvements

✅ **Rich Perception:** 16-ray raycasting provides 360° awareness
✅ **Tactical Information:** Cover detection, terrain analysis
✅ **Combat Awareness:** Weapon state, ammunition tracking
✅ **Enemy Tracking:** Top 5 nearest enemies with full details
✅ **Temporal Context:** Action history for sequence learning
✅ **Blueprint Exposed:** All fields accessible in BT tasks/services
✅ **NN-Ready:** `ToFeatureVector()` outputs normalized array

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
- No clear API boundary
- Difficult to maintain and extend


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
