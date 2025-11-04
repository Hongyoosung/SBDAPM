# Behavior Tree Structure Diagram

## Hierarchical Multi-Agent BT Architecture

```
Root [Selector]
│
├─ Services (attached to Selector, NOT root):
│  ├─ BTService_SyncCommandToBlackboard (every 0.5s)
│  └─ BTService_UpdateObservation (every 0.2s)
│
├─ [Sequence: Dead State]
│  ├─ Decorator: CheckCommandType == Dead
│  └─ Task: Wait (perpetual)
│
├─ [Sequence: Retreat Command]
│  ├─ Decorator: CheckCommandType == Retreat
│  ├─ Task: BTTask_QueryRLPolicy
│  └─ [Selector: Retreat Actions]
│     ├─ [Sequence: Evasive Movement]
│     │  ├─ Decorator: CheckTacticalAction == Evade
│     │  └─ Task: BTTask_EvasiveMovement
│     │
│     └─ [Sequence: Find Cover]
│        ├─ Decorator: CheckTacticalAction == TakeCover
│        └─ Task: BTTask_FindCoverLocation
│
├─ [Sequence: Defend Command]
│  ├─ Decorator: CheckCommandType == Defend
│  ├─ Task: BTTask_QueryRLPolicy
│  └─ Task: BTTask_ExecuteDefend
│     ├─ Internal: FindNearestCover() (tag-based)
│     ├─ Internal: Tactical positioning
│     └─ Internal: BTTask_FireWeapon integration
│
├─ [Sequence: Assault Command]
│  ├─ Decorator: CheckCommandType == Assault
│  ├─ Task: BTTask_QueryRLPolicy
│  └─ Task: BTTask_ExecuteAssault
│     ├─ Internal: Aggressive positioning
│     ├─ Internal: Enemy engagement
│     └─ Internal: BTTask_FireWeapon integration
│
├─ [Sequence: Support Command]
│  ├─ Decorator: CheckCommandType == Support
│  ├─ Task: BTTask_QueryRLPolicy
│  └─ Task: BTTask_ExecuteSupport
│     ├─ Internal: Allied positioning
│     └─ Internal: Support behavior
│
├─ [Sequence: Move Command]
│  ├─ Decorator: CheckCommandType == Move
│  ├─ Task: BTTask_QueryRLPolicy
│  └─ Task: BTTask_ExecuteMove
│     └─ Internal: NavMesh movement to target
│
└─ [Sequence: Idle/Default]
   ├─ Task: BTTask_QueryRLPolicy (optional)
   └─ Task: Wait or Patrol behavior

```

## Component Responsibilities

### Services (Attached to main Selector)
- **BTService_SyncCommandToBlackboard**: Syncs strategic command from TeamLeader → Blackboard
- **BTService_UpdateObservation**: Updates 71-feature observation vector for RL policy
- **BTService_QueryRLPolicyPeriodic**: (Alternative) Periodic RL query instead of per-branch

### Decorators
- **BTDecorator_CheckCommandType**: Filters branches by leader command (Defend, Assault, etc.)
- **BTDecorator_CheckTacticalAction**: Filters by RL-selected tactical action (Evade, TakeCover, etc.)
- **BTDecorator_CheckStrategy**: (Legacy) Strategy-based filtering

### Tasks
- **BTTask_QueryRLPolicy**: Queries RL network, writes action to blackboard
- **BTTask_ExecuteDefend**: Cover-based defensive behavior (tag-based cover)
- **BTTask_ExecuteAssault**: Aggressive enemy engagement
- **BTTask_ExecuteSupport**: Allied support positioning
- **BTTask_ExecuteMove**: NavMesh movement to command target
- **BTTask_FindCoverLocation**: EQS-based cover finding (alternative to tag-based)
- **BTTask_EvasiveMovement**: Dodge/strafe behavior
- **BTTask_FireWeapon**: Weapon firing logic
- **BTTask_SignalEventToLeader**: Notify leader of significant events
- **BTTask_UpdateTacticalReward**: RL reward signal accumulation

## Data Flow

```
Team Leader (MCTS)
    ↓ (Strategic Command)
BTService_SyncCommandToBlackboard
    ↓ (Writes to BB: "CurrentCommand")
Root Selector
    ↓ (Routes by command type)
BTDecorator_CheckCommandType
    ↓ (If matched)
BTTask_QueryRLPolicy
    ↓ (Reads 71 observations, outputs action)
    ↓ (Writes to BB: "TacticalAction")
Execute Task (Defend/Assault/Support/Move)
    ↓ (Performs action)
BTTask_SignalEventToLeader (if event occurred)
    ↓ (Notifies leader)
BTTask_UpdateTacticalReward (accumulates reward)
```

## Blackboard Keys

| Key | Type | Description |
|-----|------|-------------|
| `CurrentCommand` | Enum | Strategic command from leader (Defend, Assault, etc.) |
| `TacticalAction` | Enum | RL-selected action (Evade, TakeCover, Flank, etc.) |
| `TargetEnemy` | Actor | Current enemy target |
| `MoveToLocation` | Vector | Movement destination |
| `CoverLocation` | Vector | Selected cover position |
| `FormationPosition` | Vector | Team formation position |
| `ObservationVector` | Array | 71-element observation for RL |
| `TeamLeader` | Actor | Reference to team leader component |

## UE5 Asset Setup

**Note:** In Unreal Engine 5, services/decorators CANNOT be attached to the root node. Attach them to the first composite node (Selector) instead.

### Blueprint Setup Steps:
1. Create BehaviorTree asset (`BT_FollowerAgent`)
2. Root node → Selector (main branch selector)
3. Attach services to Selector:
   - `BTService_SyncCommandToBlackboard` (Interval: 0.5s)
   - `BTService_UpdateObservation` (Interval: 0.2s)
4. Create parallel branches for each command type:
   - Each branch starts with a Sequence node
   - Attach `BTDecorator_CheckCommandType` to each Sequence
5. Within command branches, add RL query + execution tasks
6. Configure Blackboard asset (`BB_FollowerAgent`) with keys above

### Performance Targets:
- BT tick frequency: 10-30 Hz (33-100ms)
- RL query: 1-5ms per decision
- Cover finding (tag-based): <10ms
- Total BT overhead: <0.5ms per agent

## Status

**✅ Implemented:**
- All core tasks (Execute*, Query*, Signal*, Update*)
- Command-type and tactical-action decorators
- Service infrastructure (Sync, Update)

**🔄 Needs Asset Configuration:**
- UE5 Behavior Tree asset (`BT_FollowerAgent.uasset`)
- Blackboard asset (`BB_FollowerAgent.uasset`)
- Service interval tuning
- Decorator parameter exposure

**📋 Future Enhancements:**
- BTService_QueryRLPolicyPeriodic (alternative to per-branch queries)
- BTTask_Patrol for idle behavior
- BTTask_Regroup for formation management
- EQS integration for all movement tasks (not just cover)
