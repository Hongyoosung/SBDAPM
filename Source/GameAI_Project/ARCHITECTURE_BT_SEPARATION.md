# Behavior Tree Architecture Separation

## Problem Statement

The behavior tree structure was initially shared between leader and follower agents, causing logical failures:
- `BTService_SyncCommandToBlackboard` requires `FollowerAgentComponent` (followers only)
- Leaders don't receive commands, they **issue** commands
- Leaders use MCTS for strategic planning (C++), not BT-driven tactical execution

## Solution: Role-Based BT Architecture

### Follower Agents
**Blueprint:** `BP_TestFollowerAgent`
**Behavior Tree:** `BT_FollowerAgent`
**Blackboard:** `BB_FollowerAgent`
**Component:** `UFollowerAgentComponent`

**BT Components (Follower-Only):**
- `BTService_SyncCommandToBlackboard` - Sync commands from leader
- `BTService_QueryRLPolicyPeriodic` - Periodic RL queries
- `BTTask_QueryRLPolicy` - Query tactical actions from RL policy
- `BTTask_SignalEventToLeader` - Signal events to trigger MCTS
- `BTTask_ExecuteAssault/Defend/Support/Move` - Execute tactical behaviors
- `BTDecorator_CheckCommandType` - Branch based on leader's command
- `BTDecorator_CheckTacticalAction` - Branch based on RL action

**Execution Flow:**
1. Leader issues strategic command → `UFollowerAgentComponent::ExecuteCommand()`
2. `BTService_SyncCommandToBlackboard` syncs to blackboard
3. `BTDecorator_CheckCommandType` branches BT based on command
4. `BTTask_QueryRLPolicy` queries RL for tactical action
5. Execute action subtree (assault/defend/support/move)
6. `BTTask_SignalEventToLeader` signals events back to leader

### Leader Agents
**Blueprint:** `BP_LeaderAgent`
**Behavior Tree:** **NONE** (pure C++ logic)
**Component:** `UTeamLeaderComponent`

**Execution Flow:**
1. Receive events from followers via `ProcessStrategicEvent()`
2. Event priority ≥5 triggers async MCTS planning
3. MCTS generates strategic commands
4. Issue commands to followers via `IssueCommand()`

**Why No BT?**
- Leaders use event-driven MCTS (C++), not behavior trees
- All decision logic in `TeamLeaderComponent.cpp`
- Simpler architecture, better performance
- No need for tick-based BT evaluation

## Component Requirements

| Component | Follower | Leader |
|-----------|----------|--------|
| `FollowerAgentComponent` | ✅ Required | ❌ None |
| `TeamLeaderComponent` | 🔗 Reference | ✅ Required |
| Behavior Tree | ✅ BT_FollowerAgent | ❌ None |
| RL Policy | ✅ Tactical (16 actions) | ❌ None |
| MCTS | ❌ None | ✅ Strategic |

## Migration Checklist

**✅ Completed:**
- [x] Separated follower BT assets (`BT_FollowerAgent`, `BB_FollowerAgent`)
- [x] Created leader blueprint (`BP_LeaderAgent`)
- [x] Deleted shared assets (`BT_RLTest`, `BB_RLTest`)
- [x] Documented follower-only BT components

**📋 Next Steps:**
1. Verify `BP_TestFollowerAgent` uses `BT_FollowerAgent`
2. Verify `BP_LeaderAgent` has NO behavior tree assigned
3. Test leader-follower communication in `RL_Test_Map`
4. Ensure leaders don't instantiate follower-specific BT nodes

## Code References

**Follower BT Service:**
`BTService_SyncCommandToBlackboard.cpp:43` - Queries `FollowerAgentComponent`

**Leader Communication:**
`TeamLeaderComponent.h:218` - `ProcessStrategicEvent()` entry point
`TeamLeaderComponent.h:258` - `IssueCommand()` to followers

**Follower Execution:**
`FollowerAgentComponent.h:193` - `ExecuteCommand()` from leader
`FollowerAgentComponent.h:261` - `QueryRLPolicy()` for tactical actions

## Performance Impact

**Before (Shared BT):**
- Leaders wasted cycles on follower-specific services
- Failed component queries every tick (no `FollowerAgentComponent`)
- Unnecessary blackboard updates

**After (Separated):**
- Leaders: Pure C++ event-driven (minimal overhead)
- Followers: BT + RL optimized for tactical execution
- Clean separation of concerns
