# SBDAPM: Hierarchical Multi-Agent AI System

**Engine:** Unreal Engine 5.6 | **Language:** C++17 | **Platform:** Windows

## Architecture (v2.0)

**Hierarchical Team System:** Leader (MCTS strategic) → Followers (RL tactical + BT execution)

```
Team Leader (per team) → Event-driven MCTS → Strategic commands
    ↓
Followers (N agents) → FSM + RL Policy + Behavior Tree → Tactical execution
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
- FSM transitions based on commands
- RL policy selects tactical actions
- Signals events to leader

### 3. FSM (`StateMachine.h/cpp`, `State.h/cpp`)
- **v2.0 Role:** Command-driven state transitions (NO per-agent MCTS)
- States: Idle, Assault, Defend, Support, Move, Retreat, Dead

### 4. RL Policy (`RL/RLPolicyNetwork.h/cpp`, `RL/RLReplayBuffer.h/cpp`)
- 3-layer network (128→128→64 neurons)
- PPO training algorithm
- 16 tactical actions
- Reward: +10 kill, +5 damage, -5 take damage, -10 die

### 5. Behavior Trees (`BehaviorTree/*`)
- **Tasks:** `BTTask_QueryRLPolicy`, `BTTask_ExecuteDefend`, `BTTask_ExecuteAssault`, `BTTask_ExecuteSupport`, `BTTask_ExecuteMove`, `BTTask_FindCoverLocation`, `BTTask_SignalEventToLeader`
- **Services:** `BTService_QueryRLPolicyPeriodic`, `BTService_SyncCommandToBlackboard`, `BTService_UpdateObservation`
- **Decorators:** `BTDecorator_CheckCommandType`, `BTDecorator_CheckTacticalAction`
- **Status:** ✅ Core tasks implemented for Defend/Assault/Support/Move states

### 6. EQS Cover System (`EQS/*`)
- **Generator:** `EnvQueryGenerator_CoverPoints` - Grid/tag-based cover candidate generation
- **Test:** `EnvQueryTest_CoverQuality` - Multi-factor cover scoring (enemy distance, LOS, navigability)
- **Context:** `EnvQueryContext_CoverEnemies` - Auto-fetches enemies from Team Leader
- **BT Integration:** `BTTask_FindCoverLocation` (EQS) + `BTTask_ExecuteDefend::FindNearestCover()` (tag-based)
- **Status:** ✅ Implemented, tag-based active, EQS available

### 7. Observations (`Observation/ObservationElement.h/cpp`, `TeamObservation.h/cpp`)
- **Status:** ✅ Fully updated (71 individual + 40 team features)

### 8. Communication (`Team/TeamCommunicationManager.h/cpp`)
- Leader ↔ Follower message passing
- Event priority system (triggers MCTS at priority ≥5)

### 9. Simulation Manager (`Core/SimulationManagerGameMode.h/cpp`)
- Team registration and management
- Enemy relationship tracking (mutual enemies, free-for-all)
- Actor-to-team mapping (O(1) lookup)
- **Status:** ✅ Implemented

## Current Status

**✅ Implemented:**
- Enhanced observation system (71+40 features)
- Team architecture (Leader, Follower, Communication)
- RL policy network structure (128→128→64)
- FSM refactored (command-driven, no per-agent MCTS)
- Behavior Tree core components (Tasks, Services, Decorators)
- EQS cover system (Generator, Test, Context, BT tasks)
- Simulation Manager GameMode (team registration, enemy tracking)
- Execute tasks for Defend/Assault/Support/Move states

**🔄 In Progress:**
- RL training infrastructure (experience collection, PPO updates)
- Weapon/damage system integration
- Perception system integration with Team Leader

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
- Behavior Trees execute RL-selected actions

**Performance Targets:**
- Team Leader MCTS: 50-100ms async (1-5 decisions/minute)
- Follower RL inference: 1-5ms per decision
- BT tick: <0.5ms per agent
- Total frame overhead: 10-20ms for 4-agent team

## File Structure

```
Source/GameAI_Project/
├── MCTS/              # Team leader strategic planning (event-driven)
├── RL/                # Follower tactical policies (PPO network)
├── StateMachine/      # Command-driven FSM (no per-agent MCTS)
├── BehaviorTree/      # Custom BT tasks, services, decorators
│   ├── Tasks/         # ExecuteDefend, ExecuteAssault, QueryRLPolicy, etc.
│   ├── Services/      # QueryRLPolicyPeriodic, SyncCommandToBlackboard
│   └── Decorators/    # CheckCommandType, CheckTacticalAction
├── EQS/               # Environment Query System (cover finding)
│   ├── Generator      # CoverPoints (grid + tag-based)
│   ├── Test           # CoverQuality (multi-factor scoring)
│   └── Context        # CoverEnemies (Team Leader integration)
├── Team/              # Leader, Follower, Communication
├── Observation/       # 71+40 feature observation system
└── Core/              # SimulationManagerGameMode (team management)
```

**Key Files:**
- `Team/TeamLeaderComponent.cpp` - Event-driven MCTS, strategic commands
- `Team/FollowerAgentComponent.cpp` - RL policy queries, FSM state transitions
- `BehaviorTree/Tasks/BTTask_ExecuteDefend.cpp:290-346` - Cover finding (tag-based)
- `BehaviorTree/BTTask_FindCoverLocation.cpp` - Cover finding (EQS-based)
- `EQS_SETUP_GUIDE.md` - EQS integration and setup instructions
