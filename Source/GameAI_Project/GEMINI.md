# SBDAPM: Hierarchical Multi-Agent AI System

**Engine:** Unreal Engine 5.6 | **Language:** C++17 | **Platform:** Windows

## Architecture (v2.0)

**Hierarchical Team System:** Leader (MCTS strategic) → Followers (RL tactical + StateTree execution)

```
Team Leader (per team) → Event-driven MCTS → Strategic commands
    ↓
Followers (N agents) → RL Policy + State Tree → Tactical execution
```

**Key Benefits:**
- MCTS: O(1) instead of O(n) - runs once per team, not per agent
- Event-driven: Only on significant events (enemy spotted, ally killed)
- Async: Background thread, non-blocking
- Observations: 71 features (follower) + 40 (team leader)

## Core Components

### 1. Team Leader (`Team/TeamLeaderComponent.h/cpp`)
- Event-driven MCTS (async, 500-1000 simulations)
- Issues strategic commands to followers
- Aggregates team observations (40 + N×71 features)
- Runs on background thread (50-100ms, non-blocking)

### 2. Followers (`Team/FollowerAgentComponent.h/cpp`)
- Receives commands from leader
- RL policy selects tactical actions
- Signals events to leader
- Integrates with StateTree for execution

### 3. State Tree (`StateTree/FollowerStateTreeComponent.h/cpp`)
- **v2.0 PRIMARY:** Unified execution system replacing FSM + BehaviorTree
- Command-driven state transitions (NO per-agent MCTS)
- States: Idle, Assault, Defend, Support, Move, Retreat, Dead
- **Tasks:** `STTask_QueryRLPolicy`, `STTask_ExecuteDefend`, `STTask_ExecuteAssault`, `STTask_ExecuteSupport`, `STTask_ExecuteMove`, `STTask_ExecuteRetreat`
- **Evaluators:** `STEvaluator_SyncCommand`, `STEvaluator_UpdateObservation`
- **Conditions:** `STCondition_CheckCommandType`, `STCondition_CheckTacticalAction`, `STCondition_IsAlive`
- **Status:** ✅ Implemented, replaces FSM + BehaviorTree

#### **UE 5.6 StateTree Binding Pattern (Standardized)**
All tasks/evaluators use **direct context binding** for consistency and simplicity:

**Tasks & Evaluators:**
```cpp
USTRUCT()
struct FMyTaskInstanceData {
    // AUTO-BINDS to FFollowerStateTreeContext from schema
    UPROPERTY(EditAnywhere, Category = "Context")
    FFollowerStateTreeContext Context;

    // Config properties (Parameter category)
    UPROPERTY(EditAnywhere, Category = "Parameter")
    float MyConfigValue = 1.0f;

    // Task-specific runtime state (no category)
    UPROPERTY()
    float MyRuntimeState = 0.0f;
};
```

**Conditions (lightweight - use individual bindings):**
```cpp
USTRUCT()
struct FMyConditionInstanceData {
    // Bind only the specific property needed
    UPROPERTY(EditAnywhere, Category = "Input")
    bool bIsAlive = true;
};
```

**Key Rules:**
- **Context category**: Auto-binds to matching struct name from schema (`FollowerContext`)
- **Parameter category**: Optional config values (can bind OR set in editor)
- **Input/Output categories**: For individual property bindings (conditions only)
- NO manual `FStateTreeExternalDataHandle` usage (outdated UE 5.4 pattern)
- Tasks/evaluators access shared state via `InstanceData.Context.PropertyName`

### 4. RL Policy (`RL/RLPolicyNetwork.h/cpp`, `RL/RLReplayBuffer.h/cpp`)
- 3-layer network (128→128→64 neurons)
- PPO training algorithm
- 16 tactical actions
- Reward: +10 kill, +5 damage, -5 take damage, -10 die


### 5. EQS Cover System (`EQS/*`)
- **Generator:** `EnvQueryGenerator_CoverPoints` - Grid/tag-based cover candidate generation
- **Test:** `EnvQueryTest_CoverQuality` - Multi-factor cover scoring (enemy distance, LOS, navigability)
- **Context:** `EnvQueryContext_CoverEnemies` - Auto-fetches enemies from Team Leader
- **BT Integration:** `BTTask_FindCoverLocation` (EQS) + `BTTask_ExecuteDefend::FindNearestCover()` (tag-based)
- **Status:** ✅ Implemented, tag-based active, EQS available

### 6. Observations (`Observation/ObservationElement.h/cpp`, `TeamObservation.h/cpp`)
- **Status:** ✅ Fully updated (71 individual + 40 team features)

### 7. Communication (`Team/TeamCommunicationManager.h/cpp`)
- Leader ↔ Follower message passing
- Event priority system (triggers MCTS at priority ≥5)

### 8. Perception System (`Perception/AgentPerceptionComponent.h/cpp`)
- UE5 AI Perception integration (sight-based detection)
- Team-based enemy filtering via SimulationManager
- Auto-updates RL observations with enemy data
- 360° raycasting for environmental awareness
- Auto-reports enemies to Team Leader (triggers MCTS)
- **Status:** ✅ Implemented & Validated (full pipeline tested)

### 9. Simulation Manager (`Core/SimulationManagerGameMode.h/cpp`)
- Team registration and management
- Enemy relationship tracking (mutual enemies, free-for-all)
- Actor-to-team mapping (O(1) lookup)
- **Status:** ✅ Implemented

### 10. Combat System (`Combat/HealthComponent.h/cpp`, `Combat/WeaponComponent.h/cpp`)
- **HealthComponent:** Damage/death handling, armor mitigation, health regen
- **WeaponComponent:** Configurable damage/fire rate, predictive aiming, ammo system
- **RL Integration:** Auto-binds to FollowerAgentComponent for rewards (+10 kill, +5 damage, -5 take damage, -10 die)
- **StateTree Integration:** WeaponComponent used in ExecuteAssault/Defend tasks
- **Observation Integration:** Health/ammo/cooldown auto-populated from components
- **Status:** ✅ Fully Integrated (FollowerAgentComponent.cpp:426-440, 86-102, 634-699)

## Current Status

**✅ Implemented & Validated:**
- **Command Pipeline** - Perception → Leader → MCTS (~34ms) → Commands → Followers → State Transitions ✅
- **Perception system** - Enemy detection, team filtering, auto-reporting ✅
- **Combat system** - Health/Weapon components, RL reward integration, observation population ✅
- **Comprehensive logging** - Color-coded debug system ✅
- Enhanced observation system (71+40 features)
- Team architecture (Leader, Follower, Communication)
- RL policy network structure (128→128→64)
- State Tree execution system (Tasks, Evaluators, Conditions)
- StateTree components for all follower states (Assault, Defend, Support, Move, Retreat)
- EQS cover system (Generator, Test, Context)
- Simulation Manager GameMode (team registration, enemy tracking)
- BehaviorTree (LEGACY - deprecated in favor of StateTree)

**⚠️ Current Issue:**
- **Execution Gap** - Agents receive Assault commands and transition states, but don't execute (no movement/firing)
- Root causes: No target assignment, ExecuteAssault task incomplete, possible StateTree asset gaps
- See `next_step.md` for detailed execution pipeline plan

**🔄 Next Steps (see next_step.md):**
1. **Fix ExecuteAssault Task** - Implement movement toward target + weapon firing in STTask_ExecuteAssault
2. **Target Assignment** - Team Leader assigns nearest enemy when issuing Assault commands
3. **StateTree Asset Validation** - Verify ST_FollowerBehavior has proper task bindings
4. **Movement Integration** - AIController MoveTo commands in assault execution
5. **End-to-End Combat Test** - Full loop: Perception → MCTS → Commands → Movement → Firing → Damage → Rewards

**📋 Planned:**
- Distributed training (Ray RLlib integration)
- Model persistence and loading
- Full multi-team scenarios (Red vs Blue vs Green)
- Performance profiling and optimization

## Work Instructions

**CRITICAL - Token Efficiency:**
1. **NO verbose reports** - Keep all documentation concise and code-focused
2. **NO long explanations** - Code first, brief comments only when necessary
3. **NO redundant updates** - Don't repeat what's already in this file
4. **Focus on implementation** - Spend tokens on code, not prose

**Code Style:**
- Prefer direct implementation over planning documents
- Use file:line references (e.g., `StateMachine.cpp:42`)
- Minimal comments in code unless logic is complex

**Architecture Rules:**
- Followers NEVER run MCTS (only leader does)
- All MCTS is event-driven and async
- RL policy runs per follower, not per team
- State Tree executes RL-selected actions (StateTree replaces FSM + BehaviorTree)

**Performance Targets:**
- Team Leader MCTS: 50-100ms async (1-5 decisions/minute) - **✅ ~34ms achieved**
- Follower RL inference: 1-5ms per decision
- StateTree tick: <0.5ms per agent
- Total frame overhead: 10-20ms for 4-agent team


## File Structure

```
Source/GameAI_Project/
├── MCTS/              # Team leader strategic planning (event-driven)
├── RL/                # Follower tactical policies (PPO network)
├── StateTree/         # ⭐ PRIMARY execution system
│   ├── Tasks/         # ExecuteDefend, ExecuteAssault, QueryRLPolicy, ExecuteMove, ExecuteRetreat
│   ├── Evaluators/    # SyncCommand, UpdateObservation
│   ├── Conditions/    # CheckCommandType, CheckTacticalAction, IsAlive
│   └── FollowerStateTreeComponent.h/cpp
├── Combat/            # ✅ HealthComponent, WeaponComponent (fully integrated)
├── EQS/               # Environment Query System (cover finding)
│   ├── Generator      # CoverPoints (grid + tag-based)
│   ├── Test           # CoverQuality (multi-factor scoring)
│   └── Context        # CoverEnemies (Team Leader integration)
├── Perception/        # AgentPerceptionComponent (enemy detection)
├── Team/              # Leader, Follower, Communication
├── Observation/       # 71+40 feature observation system
└── Core/              # SimulationManagerGameMode (team management)
```

**Key Files:**
- `next_step.md` - Current implementation plan (execution pipeline fixes)
- `Team/TeamLeaderComponent.cpp` - Event-driven MCTS, strategic commands
- `Team/FollowerAgentComponent.cpp` - RL observation building, combat event handling (lines 426-440, 634-699)
- `StateTree/FollowerStateTreeComponent.cpp` - Primary execution system
- `StateTree/Tasks/STTask_ExecuteDefend.cpp` - Defend state execution
- `StateTree/Tasks/STTask_ExecuteAssault.cpp` - Assault state execution with weapon firing
- `Combat/HealthComponent.cpp` - Damage/death handling, event broadcasting
- `Combat/WeaponComponent.cpp` - Weapon firing, predictive aiming
- `Perception/AgentPerceptionComponent.cpp` - Enemy detection and tracking
- `EQS_SETUP_GUIDE.md` - EQS integration and setup instructions
- `PERCEPTION_SETUP.md` - Perception system setup guide
