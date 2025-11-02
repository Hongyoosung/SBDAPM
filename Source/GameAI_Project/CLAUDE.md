# SBDAPM: State-Based Dynamic Action Planning Model - Technical Documentation

## Project Overview

**SBDAPM** (State-Based Dynamic Action Planning Model) is an advanced AI framework for Unreal Engine 5.6 that implements a hierarchical multi-agent system combining:

1. **Event-Driven MCTS** - Strategic team-level decision-making (Team Leader)
2. **Reinforcement Learning** - Tactical combat policies (Followers)
3. **Behavior Trees** - Low-level action execution
4. **Finite State Machines** - State management and coordination

**Engine:** Unreal Engine 5.6
**Language:** C++17
**Platform:** Windows (DirectX 12)
**Architecture:** Hierarchical Multi-Agent (Leader + Followers)

**📋 See [REFACTORING_PLAN.md](./REFACTORING_PLAN.md) for detailed implementation roadmap**

---

## Architecture Overview

### Hierarchical Multi-Agent System

**NEW ARCHITECTURE (v2.0):** SBDAPM uses a commander-follower pattern that separates strategic decision-making (team leader) from tactical execution (individual agents).

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNREAL ENGINE ENVIRONMENT                     │
│                  (Game World, Physics, Perception)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         │                                   │
         ▼                                   ▼
┌────────────────────┐            ┌────────────────────┐
│   TEAM LEADER      │            │   TEAM LEADER      │
│   (Red Team)       │            │   (Blue Team)      │
│                    │            │                    │
│  ┌──────────────┐  │            │  ┌──────────────┐  │
│  │ Event-Driven │  │            │  │ Event-Driven │  │
│  │    MCTS      │  │            │  │    MCTS      │  │
│  └──────────────┘  │            │  └──────────────┘  │
│                    │            │                    │
│  Input: Team Obs   │            │  Input: Team Obs   │
│  (40 + N×71 feat)  │            │  (40 + N×71 feat)  │
│                    │            │                    │
│  Output: Commands  │            │  Output: Commands  │
│  per follower      │            │  per follower      │
└─────┬──────────────┘            └──────────────┬─────┘
      │                                          │
      │ Strategic Commands                       │ Strategic Commands
      │ (Assault, Defend,                        │ (Assault, Defend,
      │  Retreat, Support)                       │  Retreat, Support)
      ▼                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FOLLOWER AGENTS (×N)                          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Agent 1      │  │ Agent 2      │  │ Agent 3      │  ...    │
│  │              │  │              │  │              │         │
│  │ ┌─────────┐  │  │ ┌─────────┐  │  │ ┌─────────┐  │         │
│  │ │   FSM   │  │  │ │   FSM   │  │  │ │   FSM   │  │         │
│  │ │(Command │  │  │ │(Command │  │  │ │(Command │  │         │
│  │ │ Driven) │  │  │ │ Driven) │  │  │ │ Driven) │  │         │
│  │ └────┬────┘  │  │ └────┬────┘  │  │ └────┬────┘  │         │
│  │      │       │  │      │       │  │      │       │         │
│  │      ▼       │  │      ▼       │  │      ▼       │         │
│  │ ┌─────────┐  │  │ ┌─────────┐  │  │ ┌─────────┐  │         │
│  │ │RL Policy│  │  │ │RL Policy│  │  │ │RL Policy│  │         │
│  │ │(Tactical│  │  │ │(Tactical│  │  │ │(Tactical│  │         │
│  │ │Actions) │  │  │ │Actions) │  │  │ │Actions) │  │         │
│  │ └────┬────┘  │  │ └────┬────┘  │  │ └────┬────┘  │         │
│  │      │       │  │      │       │  │      │       │         │
│  │      ▼       │  │      ▼       │  │      ▼       │         │
│  │ ┌─────────┐  │  │ ┌─────────┐  │  │ ┌─────────┐  │         │
│  │ │   BT    │  │  │ │   BT    │  │  │ │   BT    │  │         │
│  │ │(Execute)│  │  │ │(Execute)│  │  │ │(Execute)│  │         │
│  │ └─────────┘  │  │ └─────────┘  │  │ └─────────┘  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  Features:                                                       │
│  - Receive commands from team leader                            │
│  - FSM transitions based on commands                            │
│  - RL policy selects tactical combat actions                    │
│  - Behavior Tree executes low-level actions                     │
│  - Signal events back to leader                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Information Flow

```
1. PERCEPTION
   ↓ Follower detects event (e.g., enemy spotted)

2. EVENT SIGNAL
   ↓ Follower → Team Leader communication

3. MCTS ACTIVATION (Leader Only)
   ↓ Event-driven strategic planning (async, background thread)

4. STRATEGIC DECISION
   ↓ Leader generates commands for each follower
      - Agent 1: Assault (target enemy A)
      - Agent 2: Assault (target enemy A)
      - Agent 3: Support (cover Agent 1)
      - Agent 4: Defend (hold position)

5. COMMAND DISPATCH
   ↓ Leader → Followers communication

6. FSM TRANSITION
   ↓ Follower FSM transitions to command-appropriate state

7. RL TACTICAL QUERY
   ↓ Follower queries RL policy: "Given Assault command, which tactic?"
      Options: Aggressive, Cautious, Flanking, Suppressive

8. BEHAVIOR TREE EXECUTION
   ↓ BT executes selected tactical action
      - Pathfinding, aiming, firing, dodging

9. REWARD FEEDBACK
   ↓ BT task provides reward to RL policy (+10 for kill, -5 for damage)

10. STATUS REPORT
    ↓ Follower reports status to leader (health, ammo, progress)

11. REPEAT ON NEXT EVENT
```

### Key Architectural Benefits

| Aspect | Old System (v1.0) | New System (v2.0) |
|--------|-------------------|-------------------|
| **MCTS Execution** | Per agent (expensive) | Team leader only (efficient) |
| **Decision Frequency** | Every tick | Event-driven |
| **Coordination** | None | Explicit via leader commands |
| **Scalability** | O(n) MCTS calls | O(1) MCTS + O(n) RL inference |
| **Observation Space** | 3 features | 71 (individual) + 40 (team) |
| **Learning** | MCTS only | MCTS (strategic) + RL (tactical) |
| **Performance** | ~100ms per agent | ~100ms total for team |

---

## Core Components

### 1. Team Leader Component

**Files:** `Team/TeamLeaderComponent.h/cpp`, `Team/TeamTypes.h/cpp`

#### Purpose
Centralized strategic decision-making for a team of agents. Uses event-driven MCTS to analyze team-level observations and issue commands to followers.

#### Key Features
- **Event-Driven:** Only runs MCTS when significant events occur (enemy encounter, ally killed, etc.)
- **Asynchronous Execution:** Runs MCTS on background thread to avoid frame drops
- **Team-Level Observation:** Aggregates observations from all followers (40 base + N×71 features)
- **Command Issuance:** Sends strategic commands to individual followers
- **Follower Management:** Register/unregister followers dynamically

#### Strategic Events (Triggers MCTS)

#### Strategic Commands (MCTS Output)

#### Team Observation Structure

#### MCTS Integration


#### Performance Characteristics
- **MCTS Simulations:** 500-1000 per decision (configurable)
- **Execution Time:** 50-100ms (background thread, does not block game)
- **Decision Frequency:** Only on events (typically 1-5 per minute)
- **Scalability:** Supports 4-10 followers without performance impact

---

### 2. Follower Agent Component

**Files:** `Team/FollowerAgentComponent.h/cpp`

#### Purpose
Individual agent that executes strategic commands from the team leader using FSM + RL + Behavior Tree.

#### Key Features
- **Command-Driven FSM:** State transitions controlled by leader commands
- **RL Tactical Policy:** Selects combat tactics within strategic context
- **Behavior Tree Integration:** Executes low-level actions
- **Event Reporting:** Signals important events back to leader
- **Local Observation:** Tracks 71-feature observation for RL policy

#### Follower State Machine

#### RL Tactical Actions

#### Command Execution Flow

#### RL Policy Interface

---

### 3. Finite State Machine (FSM)

**Files:** `StateMachine.h/cpp`, `State.h/cpp`

**UPDATED ROLE:** In the new architecture, the FSM's role is simplified:
- **Command Receiver:** Transitions states based on leader commands
- **Strategic Context:** Provides context for RL policy decisions
- **State Lifecycle:** Manages EnterState/UpdateState/ExitState

**OLD ROLE (v1.0):** FSM ran MCTS per agent for every decision (inefficient)
**NEW ROLE (v2.0):** FSM transitions based on leader commands, no per-agent MCTS

#### State Lifecycle
```cpp
class UState
{
    // Called when entering this state
    virtual void EnterState(UStateMachine* StateMachine);

    // Called every frame while in this state
    virtual void UpdateState(UStateMachine* StateMachine, float DeltaTime);

    // Called when exiting this state
    virtual void ExitState(UStateMachine* StateMachine);
};
```

#### States (Follower Agents)
1. **IdleState:** No active command
2. **AssaultState:** Executing offensive command
3. **DefendState:** Executing defensive command
4. **SupportState:** Executing support command
5. **MoveState:** Executing movement command
6. **RetreatState:** Executing retreat command
7. **DeadState:** Terminal state

#### State-Command Mapping

---

### 4. Event-Driven MCTS (Team Leader)

**Files:** `MCTS/MCTS.h/cpp`, `MCTS/MCTSNode.h/cpp`

**UPDATED ARCHITECTURE:** MCTS now operates at team level, not individual agent level.

#### Purpose
Strategic planning for team-level decisions. Explores possible command combinations for all followers and selects the optimal team strategy.

#### Key Differences from v1.0

| Aspect | v1.0 (Per-Agent MCTS) | v2.0 (Team-Level MCTS) |
|--------|----------------------|------------------------|
| **Observation** | 3 features (Health, Distance, Enemies) | 40 + N×71 features (team + followers) |
| **Action Space** | Individual actions | Command assignments per follower |
| **Frequency** | Every tick | Event-driven (1-5 per minute) |
| **Execution** | Synchronous | Asynchronous (background thread) |
| **Reward** | Individual agent reward | Team-level reward |

#### MCTS Action Space (Team Level)
```cpp
// Example: 4 followers, 15 command types
// Action Space Size = 15^4 = 50,625 possible command combinations
//
// MCTS explores this space to find optimal team strategy:
// - Follower 1: Assault → Enemy A
// - Follower 2: Flank → Enemy A (from left)
// - Follower 3: Support → Follower 1 (covering fire)
// - Follower 4: Defend → Hold position
```


#### Async Execution



### 5. Reinforcement Learning (RL) Policy

**Files:** `RL/RLPolicyNetwork.h/cpp`, `RL/RLReplayBuffer.h/cpp`

**NEW COMPONENT:** Neural network-based RL policy for tactical decision-making by follower agents.

#### Purpose
Each follower agent uses an RL policy to select tactical actions within the strategic context provided by the team leader.

#### Policy Architecture
```
Input Layer (71 features)
     ↓
Hidden Layer 1 (128 neurons, ReLU)
     ↓
Hidden Layer 2 (128 neurons, ReLU)
     ↓
Hidden Layer 3 (64 neurons, ReLU)
     ↓
Output Layer (16 actions, Softmax)
```

#### Training Algorithm
**Proximal Policy Optimization (PPO)**
- **Advantages:** Stable, sample-efficient, works well with continuous learning
- **Batch Size:** 32-64 experiences
- **Learning Rate:** 0.0003
- **Discount Factor:** 0.99
- **Clipping Parameter:** 0.2

#### Experience Replay


#### Reward Structure (Follower-Level)
```cpp
// Combat rewards
+10  Kill enemy
+5   Damage enemy
+3   Suppress enemy
-5   Take damage
-10  Die

// Tactical rewards
+5   Successfully reach cover
+3   Maintain formation
+2   Follow command effectively
-3   Break formation
-5   Ignore command

// Support rewards
+10  Rescue ally
+5   Provide covering fire
+3   Share ammunition
```

---

### 6. Behavior Tree System (Tactical Layer)

**Files:** `BehaviorTree/*`, Custom tasks/services/decorators in `BehaviorTree/` subdirectories

#### Purpose
Execute low-level actions based on FSM state and RL tactical action. Handles pathfinding, combat mechanics, animation, and environment interaction.

#### Key Behavior Tree Components

**Custom Tasks:**
```cpp
UBTTask_QueryRLPolicy          // Query RL policy for tactical action
UBTTask_SignalEventToLeader    // Signal event to team leader
UBTTask_FireWeapon             // Execute weapon fire
UBTTask_FindCover              // Find cover using EQS
UBTTask_UpdateTacticalReward   // Provide reward to RL policy
UBTTask_EvasiveMovement        // Perform evasive maneuvers
```

**Custom Services:**
```cpp
UBTService_UpdateObservation          // Gather observation data
UBTService_SyncCommandToBlackboard    // Sync leader command to blackboard
UBTService_QueryRLPolicyPeriodic      // Periodically query RL policy
```

**Custom Decorators:**
```cpp
UBTDecorator_CheckCommandType      // Check if command matches required type
UBTDecorator_CheckTacticalAction   // Check if RL action matches required action
```

#### Behavior Tree Structure
```
Root (Selector)
│
├─ [Decorator: CommandType == "Dead"] DeadBehavior
│  └─ Task: PlayDeathAnimation + SignalDeathToLeader
│
├─ [Decorator: CommandType == "Retreat"] RetreatBehavior
│  ├─ [TacticalAction == Sprint] SprintRetreatSubtree
│  └─ [TacticalAction == SeekCover] CoveredRetreatSubtree
│
├─ [Decorator: CommandType == "Assault"] AssaultBehavior
│  ├─ [TacticalAction == AggressiveAssault] AggressiveSubtree
│  ├─ [TacticalAction == CautiousAdvance] CautiousSubtree
│  ├─ [TacticalAction == FlankLeft] FlankLeftSubtree
│  └─ [TacticalAction == FlankRight] FlankRightSubtree
│
├─ [Decorator: CommandType == "Defend"] DefendBehavior
│  ├─ [TacticalAction == DefensiveHold] DefensiveHoldSubtree
│  ├─ [TacticalAction == SeekCover] SeekCoverSubtree
│  └─ [TacticalAction == SuppressiveFire] SuppressiveFireSubtree
│
├─ [Decorator: CommandType == "Support"] SupportBehavior
│  ├─ [Decorator: AllyNeedsRescue?] RescueSubtree
│  └─ [Decorator: AllyEngaged?] CoveringFireSubtree
│
└─ [Decorator: CommandType == "MoveTo"] MoveToBehavior
   ├─ [TacticalAction == Sprint] SprintMoveSubtree
   ├─ [TacticalAction == Crouch] StealthMoveSubtree
   └─ [Default] NormalMoveSubtree
```

---

### 7. Communication System

**Files:** `Team/TeamCommunicationManager.h/cpp`

#### Purpose
Manages message passing between team leader and followers, and between followers (peer-to-peer).

#### Message Types

**Leader → Follower:**
```cpp
SendCommandToFollower()        // Issue strategic command
SendFormationUpdate()          // Update formation positions
SendAcknowledgement()          // Acknowledge event received
```

**Follower → Leader:**
```cpp
SignalEventToLeader()          // Report strategic event
ReportCommandComplete()        // Report command completion
RequestAssistance()            // Request help (low health, ammo, etc.)
ReportStatus()                 // Periodic status update
```

**Follower ↔ Follower (Optional):**
```cpp
SendPeerMessage()              // Send message to specific teammate
BroadcastToNearby()            // Broadcast to nearby teammates (radius)
```

#### Event Priority System
```cpp
enum class EEventPriority : uint8
{
    Low = 1,        // Minor events (e.g., waypoint reached)
    Medium = 5,     // Normal events (e.g., enemy spotted)
    High = 8,       // Important events (e.g., under fire)
    Critical = 10   // Critical events (e.g., ally killed, ambush)
};

// Leader MCTS triggered when:
// - Event priority >= EventPriorityThreshold (default: 5)
// - OR event type is in critical list
// - AND MCTS not already running
// - AND cooldown expired (default: 2 seconds)
```

---

### 8. Observation System

**Files:** `Observation/ObservationElement.h/cpp`, `Observation/TeamObservation.h/cpp`

**STATUS:** ✅ **FULLY UPDATED** - Now provides 71 features (individual) + 40 features (team)

#### Individual Agent Observation (71 Features)

```cpp
struct FObservationElement
{
    // AGENT STATE (12 features)
    FVector Position, Velocity;
    FRotator Rotation;
    float Health, Stamina, Shield;

    // COMBAT STATE (3 features)
    float WeaponCooldown, Ammunition;
    int32 CurrentWeaponType;

    // ENVIRONMENT PERCEPTION (32 features)
    TArray<float> RaycastDistances;          // 16 rays, 360° coverage
    TArray<ERaycastHitType> RaycastHitTypes; // What each ray detected

    // ENEMY INFORMATION (16 features)
    int32 VisibleEnemyCount;
    TArray<FEnemyObservation> NearbyEnemies; // Top 5 closest (dist, health, angle)

    // TACTICAL CONTEXT (5 features)
    bool bHasCover;
    float NearestCoverDistance;
    FVector2D CoverDirection;
    ETerrainType CurrentTerrain;

    // TEMPORAL FEATURES (2 features)
    float TimeSinceLastAction;
    int32 LastActionType;

    // LEGACY (1 feature)
    float DistanceToDestination;

    // Utility
    TArray<float> ToFeatureVector() const;  // Returns 71 normalized values
};
```

#### Team Observation (40 Base + N×71 Features)

See "Team Leader Component" section above for full structure.

#### Observation Flow

**1. Gathering (Behavior Tree Service):**
```cpp
void UBTService_UpdateObservation::TickNode(...)
{
    FObservationElement Observation;

    // Gather agent state
    Observation.Position = Pawn->GetActorLocation();
    Observation.Velocity = Pawn->GetVelocity();
    Observation.Health = HealthComponent->GetHealthPercentage();

    // Perform raycasts
    PerformEnvironmentRaycasts(Pawn, Observation);

    // Detect enemies
    DetectNearbyEnemies(Pawn, Observation);

    // Update follower component
    FollowerComponent->UpdateLocalObservation(Observation);
}
```

**2. Usage (RL Policy):**
```cpp
ETacticalAction URLPolicyNetwork::SelectAction(const FObservationElement& Obs)
{
    // Convert to feature vector
    TArray<float> Features = Obs.ToFeatureVector();  // 71 values

    // Forward pass through neural network
    TArray<float> ActionProbabilities = ForwardPass(Features);

    // Select action (epsilon-greedy or softmax)
    ETacticalAction Action = SampleAction(ActionProbabilities);

    return Action;
}
```

**3. Aggregation (Team Leader):**
```cpp
FTeamObservation UTeamLeaderComponent::BuildTeamObservation()
{
    FTeamObservation TeamObs;

    // Gather follower observations
    for (AActor* Follower : GetAliveFollowers())
    {
        UFollowerAgentComponent* Component = Follower->FindComponentByClass(...);
        TeamObs.FollowerObservations.Add(Component->GetLocalObservation());
    }

    // Calculate team metrics
    TeamObs.AverageTeamHealth = CalculateAverageHealth();
    TeamObs.TeamCentroid = CalculateTeamCentroid();
    TeamObs.FormationCoherence = CalculateFormationCoherence();
    // ... etc

    return TeamObs;
}
```

---

## System Comparison

### v1.0 (Old System) vs v2.0 (New Hierarchical System)

| Feature | v1.0 | v2.0 |
|---------|------|------|
| **Architecture** | Flat (per-agent MCTS) | Hierarchical (leader + followers) |
| **MCTS** | Every agent, every tick | Team leader, event-driven |
| **Observations** | 3 features | 71 (individual) + 40 (team) |
| **Learning** | MCTS only | MCTS (strategic) + RL (tactical) |
| **Coordination** | None | Explicit commands |
| **Scalability** | Poor (O(n) MCTS) | Excellent (O(1) MCTS) |
| **Performance** | ~100ms per agent | ~100ms total |
| **Tactical AI** | Blueprint events | RL policy + Behavior Tree |
| **State Management** | FSM with MCTS | FSM (command-driven) |
| **Extensibility** | Difficult | Modular |

---

## Critical Issues & Improvements

### ✅ Resolved in v2.0

1. **✅ Per-Agent MCTS Overhead** → Event-driven team leader MCTS
2. **✅ Limited Observation Space** → 71 features (individual) + 40 (team)
3. **✅ No Coordination** → Explicit leader commands
4. **✅ Synchronous Execution** → Asynchronous MCTS on background thread
5. **✅ Poor Scalability** → Hierarchical architecture supports 10+ agents

### 🔄 In Progress

6. **🔄 Neural Network Integration** → RL policy implemented, needs training infrastructure
7. **🔄 Behavior Tree Components** → Designed, needs implementation
8. **🔄 Complete State Implementations** → FleeState/DeadState need work

### 📋 Planned

9. **📋 Distributed Training** → Ray RLlib integration (Phase 6)
10. **📋 Model Persistence** → Save/load trained policies (Phase 5)
11. **📋 Multi-Team Support** → Red vs Blue teams (Phase 6)
12. **📋 Explainability Tools** → MCTS tree visualization (Phase 6)

---

## Implementation Roadmap

**See [REFACTORING_PLAN.md](./REFACTORING_PLAN.md) for detailed implementation plan**

### Summary Timeline (24 Weeks)

- **Weeks 1-3:** Foundation (code restructuring, enhanced observations, complete states)
- **Weeks 4-7:** Team Architecture (team leader, followers, communication)
- **Weeks 8-11:** Reinforcement Learning (RL policies, training, integration)
- **Weeks 12-15:** Behavior Tree (custom tasks, services, decorators, assets)
- **Weeks 16-18:** Integration & Testing (end-to-end, performance, bug fixes)
- **Weeks 19-22:** Advanced Features (distributed training, multi-team, explainability)
- **Weeks 23-24:** Documentation & Release

---

## Usage Example

### Setting Up a Team

```cpp
void AMyGameMode::BeginPlay()
{
    Super::BeginPlay();

    // Create team leader
    AActor* LeaderActor = GetWorld()->SpawnActor<AActor>();
    UTeamLeaderComponent* Leader = NewObject<UTeamLeaderComponent>(LeaderActor);
    Leader->RegisterComponent();
    Leader->TeamName = TEXT("Alpha Team");
    Leader->MaxFollowers = 4;
    Leader->bAsyncMCTS = true;
    Leader->MCTSSimulations = 500;

    // Spawn followers
    TArray<AActor*> Followers;
    for (int32 i = 0; i < 4; i++)
    {
        AActor* Follower = GetWorld()->SpawnActor<AGameAICharacter>(
            AGameAICharacter::StaticClass(),
            FVector(i * 200.0f, 0, 0),
            FRotator::ZeroRotator
        );

        // Add follower component
        UFollowerAgentComponent* FollowerComp = NewObject<UFollowerAgentComponent>(Follower);
        FollowerComp->RegisterComponent();
        FollowerComp->TeamLeader = Leader;

        // Register with leader
        Leader->RegisterFollower(Follower);

        Followers.Add(Follower);
    }

    // Bind to events
    Leader->OnStrategicDecisionMade.AddDynamic(this, &AMyGameMode::OnTeamDecisionMade);
}

void AMyGameMode::OnTeamDecisionMade(const TMap<AActor*, FStrategicCommand>& Commands)
{
    UE_LOG(LogTemp, Log, TEXT("Team issued %d commands"), Commands.Num());
}
```

### Triggering Strategic Events

```cpp
void AGameAICharacter::OnEnemySpotted(AActor* Enemy)
{
    // Get follower component
    UFollowerAgentComponent* Follower = FindComponentByClass<UFollowerAgentComponent>();
    if (!Follower) return;

    // Signal event to team leader
    Follower->SignalEventToLeader(
        EStrategicEvent::EnemyEncounter,
        Enemy
    );

    // Leader will process event and may trigger MCTS
    // Follower will receive new command and execute via RL + BT
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

## Performance Characteristics

### Expected Performance (4-Agent Team)

| Component | Execution Time | Frequency |
|-----------|---------------|-----------|
| Team Leader MCTS | 50-100ms | Event-driven (1-5/min) |
| Follower RL Inference | 1-5ms | Per decision (~1/sec) |
| Behavior Tree Tick | 0.5ms | Every frame |
| Observation Gathering | 2-5ms | Every frame |
| **Total Overhead** | **~10-20ms** | **Per frame** |

### Scalability

| Team Size | MCTS Time | Total RL Time | Frame Impact |
|-----------|-----------|---------------|--------------|
| 2 agents | 50ms | 2-10ms | ~12ms |
| 4 agents | 75ms | 4-20ms | ~24ms |
| 8 agents | 100ms | 8-40ms | ~48ms |
| 16 agents | 150ms | 16-80ms | ~96ms |

**Note:** MCTS runs asynchronously, does not block frame. RL and BT run on game thread.

