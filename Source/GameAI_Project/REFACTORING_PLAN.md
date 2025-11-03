# SBDAPM Refactoring Plan v2.0

**Status:** In Progress | **Target:** 24 weeks

## Key Changes

| Aspect | v1.0 | v2.0 |
|--------|------|------|
| Architecture | Per-agent MCTS | Hierarchical (leader + followers) |
| MCTS | Every agent, every tick | Event-driven, team leader only |
| Tactical AI | Blueprint events | RL policy + BT |
| Observations | 3 features | 71 (individual) + 40 (team) |
| Scalability | O(n) | O(1) MCTS + O(n) RL |

## Core Modules

### 1. Enhanced Observations
- **Files:** `Observation/ObservationElement.h/cpp`, `TeamObservation.h/cpp`
- **Status:** ✅ Implemented
- 71 features (individual), 40 features (team)

### 2. Team Leader
- **Files:** `Team/TeamLeaderComponent.h/cpp`, `TeamTypes.h/cpp`
- **Status:** 🔄 Foundation complete, MCTS integration needed
- Event-driven MCTS (async), issues strategic commands

### 3. Follower Agent
- **Files:** `Team/FollowerAgentComponent.h/cpp`
- **Status:** 🔄 Foundation complete
- Receives commands, runs RL policy, executes via BT

### 4. Communication
- **Files:** `Team/TeamCommunicationManager.h/cpp`
- **Status:** 📋 Planned
- Leader ↔ Follower, event priority system

### 5. RL Policy
- **Files:** `RL/RLPolicyNetwork.h/cpp`, `RLReplayBuffer.h/cpp`
- **Status:** 🔄 Network structure complete, training needed
- 3-layer network, PPO training, 16 tactical actions

### 6. Behavior Trees
- **Files:** `BehaviorTree/Tasks/`, `Services/`, `Decorators/`
- **Status:** 📋 Planned
- Custom tasks: QueryRLPolicy, SignalEvent, FireWeapon, FindCover
- Custom services: UpdateObservation, SyncCommand
- Custom decorators: CheckCommandType, CheckTacticalAction

### 7. Event-Driven MCTS
- **Files:** `MCTS/MCTS.h/cpp` (refactored)
- **Status:** 🔄 Per-agent version exists, needs team-level adaptation
- Team action space, async execution, event triggers

## Implementation Phases

### Phase 1: Foundation (Weeks 1-3) ✅
- Code restructuring
- Enhanced observations (71+40 features)
- Complete existing states

### Phase 2: Team Architecture (Weeks 4-7) 🔄
- Team Leader component
- Follower component
- Communication manager
- Command system

### Phase 3: RL Integration (Weeks 8-11) 📋
- RL policy network
- Training infrastructure
- Replay buffer
- Reward system

### Phase 4: Behavior Trees (Weeks 12-15) 📋
- Custom BT tasks
- Custom BT services
- Custom BT decorators
- BT assets creation

### Phase 5: Integration (Weeks 16-18) 📋
- End-to-end testing
- Performance optimization
- Bug fixes

### Phase 6: Advanced (Weeks 19-22) 📋
- Distributed training (Ray RLlib)
- Multi-team support
- Model persistence
- MCTS visualization

### Phase 7: Release (Weeks 23-24) 📋
- Documentation
- Polish
- Release prep

## Critical Work Instructions

**For ALL future work:**
1. **NO detailed planning documents** - Write code directly
2. **NO verbose reports** - Single line status updates only
3. **NO redundant documentation** - Update this file ONLY when architecture changes
4. **Code-focused responses** - Spend tokens on implementation, not explanation
5. **Use file:line references** - e.g., `TeamLeaderComponent.cpp:156`

## Current Priorities

1. **TeamLeaderComponent:** Complete MCTS integration
2. **FollowerAgentComponent:** Wire up to FSM and BT
3. **Communication:** Implement message passing
4. **BT Tasks:** Start with UBTTask_QueryRLPolicy

## Performance Targets

- Team MCTS: 50-100ms async
- RL inference: 1-5ms per agent
- BT tick: <0.5ms per agent
- Frame: 60 FPS stable (8 agents)

## File Structure

```
Source/GameAI_Project/
├── MCTS/              # Team strategic planning
├── RL/                # Follower tactical policies
├── StateMachine/      # Command-driven FSM (NO per-agent MCTS)
├── BehaviorTree/      # Custom components
├── Team/              # Leader, Follower, Communication
└── Observation/       # 71+40 features
```

## Testing Checklist

- [ ] Team leader spawns and registers followers
- [ ] Events trigger MCTS (async, non-blocking)
- [ ] Commands sent from leader to followers
- [ ] Followers transition states based on commands
- [ ] RL policy selects tactical actions
- [ ] BT executes actions
- [ ] Rewards flow to RL policy
- [ ] 4v4 combat works end-to-end

## Migration Strategy

1. **Weeks 1-3:** New code alongside old ✅
2. **Weeks 4-11:** Parallel systems (toggle-able) 🔄
3. **Weeks 12-15:** Migrate to new system
4. **Weeks 16-18:** Remove old code
5. **Weeks 19-24:** Polish and release

---

**Remember:** Code first, documentation never (unless architecture changes). Keep responses minimal and focused on implementation.
