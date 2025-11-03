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
- Custom tasks: `UBTTask_QueryRLPolicy`, `UBTTask_SignalEventToLeader`, etc.
- Custom services: `UBTService_UpdateObservation`, etc.
- Custom decorators: `UBTDecorator_CheckCommandType`, etc.

### 6. Observations (`Observation/ObservationElement.h/cpp`, `TeamObservation.h/cpp`)
- **Status:** ✅ Fully updated (71 individual + 40 team features)

### 7. Communication (`Team/TeamCommunicationManager.h/cpp`)
- Leader ↔ Follower message passing
- Event priority system (triggers MCTS at priority ≥5)

## Current Status

**✅ Implemented:**
- Enhanced observation system (71+40 features)
- Team architecture foundations
- RL policy network structure
- FSM refactored (command-driven, no per-agent MCTS)

**🔄 In Progress:**
- Behavior Tree custom components
- RL training infrastructure
- Complete state implementations

**📋 Planned:**
- Distributed training (Ray RLlib)
- Model persistence
- Multi-team support (Red vs Blue)

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
├── MCTS/              # Team leader strategic planning
├── RL/                # Follower tactical policies
├── StateMachine/      # Command-driven FSM
├── BehaviorTree/      # Custom BT components
├── Team/              # Leader, Follower, Communication
└── Observation/       # 71+40 feature observation system
```

**See REFACTORING_PLAN.md for detailed roadmap**
