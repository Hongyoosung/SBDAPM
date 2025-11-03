# Behavior Tree Asset Creation Guide

**Quick Reference for SBDAPM Behavior Tree Setup**
**Date:** 2025-11-02

---

## Quick Setup (5 Minutes)

### 1. Create Blackboard Asset

**Steps:**
1. Content Browser → Right-click → AI → **Blackboard**
2. Name: `BB_FollowerAgent`
3. Double-click to open
4. Add the following keys:

| Key Name | Type | Default Value |
|----------|------|---------------|
| `CommandType` | Enum → `EStrategicCommandType` | Idle |
| `CommandTarget` | Object → `AActor` | None |
| `CommandPriority` | Int | 5 |
| `TimeSinceCommand` | Float | 0.0 |
| `IsCommandValid` | Bool | false |
| `TacticalAction` | Enum → `ETacticalAction` | DefensiveHold |
| `ActionProgress` | Float | 0.0 |
| `TargetActor` | Object → `AActor` | None |
| `DefendLocation` | Vector | (0,0,0) |
| `MoveDestination` | Vector | (0,0,0) |

5. Save and close

---

### 2. Create Behavior Tree Asset

**Steps:**
1. Content Browser → Right-click → AI → **Behavior Tree**
2. Name: `BT_FollowerAgent`
3. Double-click to open
4. **Properties Panel:**
   - Blackboard Asset: `BB_FollowerAgent`

---

### 3. Configure Root Node (Essential Services)

**IMPORTANT:** Services cannot be attached to the root node in UE5. You must attach them to a composite node.

**Root Structure:**
1. Right-click Root → Add Composite → **Selector** (main selector)

2. Right-click Main Selector → Add Service → **SyncCommandToBlackboard**
   - Interval: `0.5`
   - Random Deviation: `0.1`
   - CommandTypeKey: `CommandType`
   - CommandTargetKey: `CommandTarget`
   - CommandPriorityKey: `CommandPriority`
   - TimeSinceCommandKey: `TimeSinceCommand`
   - IsCommandValidKey: `IsCommandValid`
   - bClearOnNoFollowerComponent: `true`
   - bLogSync: `false` (enable for debugging)

---

### 4. Add Command Branches

**Available Strategic Command Types (TeamTypes.h:44-79):**
- **Offensive:** Assault, Flank, Suppress, Charge
- **Defensive:** StayAlert, HoldPosition, TakeCover, Fortify
- **Support:** RescueAlly, ProvideSupport, Regroup, ShareAmmo
- **Movement:** Advance, Retreat, Patrol, MoveTo, Follow
- **Special:** Investigate, Distract, Stealth, Idle

**Under Main Selector:**

#### Offensive Branch (Example)

1. **Add Composite: Selector** (Offensive Branch)

2. **Add Decorator: CheckCommandType**
   - AcceptedCommandTypes: `[Assault, Flank, Charge, Suppress]` (group offensive commands)
   - bUseBlackboard: `true`
   - CommandTypeKey: `CommandType`
   - bRequireValidCommand: `true`
   - IsCommandValidKey: `IsCommandValid`
   - **Flow Abort Mode:** `Lower Priority`

3. **Add Service: QueryRLPolicyPeriodic**
   - Interval: `1.0`
   - Random Deviation: `0.2`
   - TacticalActionKey: `TacticalAction`
   - bEnableExploration: `true`
   - bRequireActiveCommand: `false`
   - bLogQueries: `false` (enable for debugging)

4. **Add Child Selector** (Tactical Action Branches)

5. **For Each Tactical Action:**

   **Aggressive Assault:**
   - Composite: Sequence
   - Decorator: CheckTacticalAction
     - AcceptedActions: `[AggressiveAssault]`
     - TacticalActionKey: `TacticalAction`
     - **Flow Abort Mode:** `Self`
   - Task: **BTTask_ExecuteAssault**
     - (Configure assault parameters as needed)

   **Cautious Advance:**
   - Composite: Sequence
   - Decorator: CheckTacticalAction
     - AcceptedActions: `[CautiousAdvance]`
     - TacticalActionKey: `TacticalAction`
     - **Flow Abort Mode:** `Self`
   - Task: **BTTask_ExecuteAssault**
     - (Configure for cautious approach)

   **Flanking:**
   - Composite: Sequence
   - Decorator: CheckTacticalAction
     - AcceptedActions: `[FlankLeft, FlankRight]`
     - TacticalActionKey: `TacticalAction`
     - **Flow Abort Mode:** `Self`
   - Task: **BTTask_ExecuteAssault**
     - (Configure for flanking maneuver)

---

### 5. Add Other Command Branches (Same Pattern)

**Defensive Branch:**
- Decorator: CheckCommandType → `[StayAlert, HoldPosition, TakeCover, Fortify]`
- Service: QueryRLPolicyPeriodic (Interval: 1.5s)
- Tactical branches:
  - DefensiveHold → BTTask_ExecuteDefend
  - SeekCover → BTTask_ExecuteDefend
  - SuppressiveFire → BTTask_ExecuteDefend

**Support Branch:**
- Decorator: CheckCommandType → `[RescueAlly, ProvideSupport, Regroup, ShareAmmo]`
- Service: QueryRLPolicyPeriodic (Interval: 1.0s)
- Tactical branches:
  - ProvideCoveringFire → BTTask_ExecuteSupport
  - Reload → BTTask_ExecuteSupport
  - RescueAlly → BTTask_ExecuteSupport

**Movement Branch:**
- Decorator: CheckCommandType → `[Advance, Retreat, Patrol, MoveTo, Follow]`
- Service: QueryRLPolicyPeriodic (Interval: 1.5s)
- Tactical branches:
  - Sprint → BTTask_ExecuteMove
  - Crouch → BTTask_ExecuteMove
  - Patrol → BTTask_ExecuteMove

**Idle Branch (Default - REQUIRED):**
- Decorator: CheckCommandType → `[Idle]` OR none (fallback branch)
- Task: Wait or Idle Animation
- **Note:** This branch handles initial state before team leader issues commands. Without it, behavior tree will fail if no command is active (IsCommandValid = false).

---

## Visual Structure

```
Root
└─ Main Selector
   ├─ [Service: SyncCommandToBlackboard @ 0.5s]
   │
   ├─ [CheckCommandType: Assault|Flank|Charge|Suppress, FlowAbort: LowerPriority]
   │  └─ Selector (Offensive Subtree)
   │     ├─ [Service: QueryRLPolicyPeriodic @ 1.0s]
   │     │
   │     ├─ [CheckTacticalAction: AggressiveAssault, FlowAbort: Self]
   │     │  └─ Sequence → BTTask_ExecuteAssault (Aggressive)
   │     │
   │     ├─ [CheckTacticalAction: CautiousAdvance, FlowAbort: Self]
   │     │  └─ Sequence → BTTask_ExecuteAssault (Cautious)
   │     │
   │     ├─ [CheckTacticalAction: FlankLeft|FlankRight, FlowAbort: Self]
   │     │  └─ Sequence → BTTask_ExecuteAssault (Flanking)
   │     │
   │     └─ [Default] → BTTask_ExecuteAssault (Generic)
   │
   ├─ [CheckCommandType: StayAlert|HoldPosition|TakeCover|Fortify, FlowAbort: LowerPriority]
   │  └─ Selector (Defensive Subtree)
   │     ├─ [Service: QueryRLPolicyPeriodic @ 1.5s]
   │     ├─ [Tactical Action Branches...]
   │     └─ BTTask_ExecuteDefend
   │
   ├─ [CheckCommandType: RescueAlly|ProvideSupport|Regroup|ShareAmmo, FlowAbort: LowerPriority]
   │  └─ Selector (Support Subtree) → BTTask_ExecuteSupport
   │
   ├─ [CheckCommandType: Advance|Retreat|Patrol|MoveTo|Follow, FlowAbort: LowerPriority]
   │  └─ Selector (Movement Subtree) → BTTask_ExecuteMove
   │
   └─ [CheckCommandType: Idle OR No Decorator - Fallback]
      └─ Wait Task (handles initial state & no active commands)
```

---

## Common Patterns

### Pattern 1: Command-Based Branching

```
Selector
├─ [CheckCommandType: CommandA] → Subtree A
├─ [CheckCommandType: CommandB] → Subtree B
└─ [Default] → Idle Behavior
```

**Key Settings:**
- FlowAbortMode: `Lower Priority` (abort lower branches when command changes)
- bUseBlackboard: `true`
- CommandTypeKey: `CommandType`

---

### Pattern 2: Tactical Action Branching

```
Command Subtree (e.g., Assault)
├─ [Service: QueryRLPolicyPeriodic]
│
├─ [CheckTacticalAction: ActionA] → Execute ActionA
├─ [CheckTacticalAction: ActionB] → Execute ActionB
└─ [Default] → Generic Execution
```

**Key Settings:**
- FlowAbortMode: `Self` (abort self when action changes)
- TacticalActionKey: `TacticalAction`
- Service Interval: 1.0-1.5s

---

### Pattern 3: Parallel Services

```
Root Composite
├─ [Service: SyncCommandToBlackboard]       (Syncs command data)
│
└─ Subtree
   ├─ [Service: QueryRLPolicyPeriodic]     (Queries RL policy)
   └─ [Task Execution...]
```

**Why?**
- SyncCommandToBlackboard runs at root (affects all branches)
- QueryRLPolicyPeriodic runs per command (only when needed)

---

## Flow Abort Modes Explained

| Mode | When to Use | Example |
|------|-------------|---------|
| **None** | No reactivity needed | Idle behaviors |
| **Self** | Abort self when condition fails | Tactical action branches |
| **Lower Priority** | Abort lower branches when condition becomes true | Command type branches |
| **Both** | Maximum reactivity | Critical high-priority behaviors |

**Best Practices:**
- Command branches: `Lower Priority` (switch commands immediately)
- Tactical branches: `Self` (switch tactics when RL changes action)
- Tasks: Usually no decorators (controlled by parent)

---

## Service Intervals Guide

| Service | Recommended Interval | Why |
|---------|---------------------|-----|
| SyncCommandToBlackboard | 0.5s | Commands change infrequently, no need for faster sync |
| QueryRLPolicyPeriodic (Assault) | 1.0s | Fast-paced combat, needs frequent updates |
| QueryRLPolicyPeriodic (Defend) | 1.5s | Slower pace, defensive stance changes less |
| QueryRLPolicyPeriodic (Support) | 1.0s | Needs to respond quickly to ally status |
| QueryRLPolicyPeriodic (Move) | 1.5s | Movement is gradual, no need for fast queries |

**For Large Teams (8+ agents):**
- Increase all intervals by 50-100% (e.g., 1.0s → 1.5-2.0s)
- Enable `bQueryOnlyWhenObservationChanged` on QueryRLPolicyPeriodic

---

## Debugging Tips

### Enable Logging

**In BT Services:**
```
SyncCommandToBlackboard:
  bLogSync = true

QueryRLPolicyPeriodic:
  bLogQueries = true
```

**In BT Decorators:**
```
CheckTacticalAction:
  bLogChecks = true
```

### Visual Debugging

1. **Play in Editor (PIE)**
2. **Select AI agent in World Outliner**
3. **Gameplay Debugger:** Press `'` (apostrophe) key
4. **BT Debug View:** Shows active nodes, blackboard values, services

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| BT doesn't execute | No AI Controller running BT | Check AIController::BeginPlay() calls RunBehaviorTree() |
| Decorators always fail | Blackboard key not set | Verify service is updating the key |
| **Service returns false, blocking subtree** | **No active command from team leader** | **Add Idle fallback branch at bottom of selector. IsCommandValid=false when CommandType=Idle or no command received yet (FollowerAgentComponent.cpp:237)** |
| Services don't run | Interval too long or not ticking | Check Interval > 0, service added to composite |
| RL policy not queried | TacticalPolicy not initialized | Check FollowerAgentComponent has TacticalPolicy set |
| Command changes ignored | Flow abort mode wrong | Use `Lower Priority` for command decorators |
| Service cannot be placed on root | UE5 limitation | Attach service to first composite node (Selector/Sequence) under root |
| Missing CommandPriority/TimeSinceCommand keys | Guide was incomplete | Add Int key `CommandPriority` (default: 5) and Float key `TimeSinceCommand` (default: 0.0) to blackboard |

---

## Blueprint Integration

### Running BT from Blueprint

**AI Controller Blueprint:**

1. **Event BeginPlay**
2. **Run Behavior Tree**
   - Behavior Tree: `BT_FollowerAgent`
3. **Use Blackboard**
   - Blackboard Asset: `BB_FollowerAgent`

**Pawn/Character Blueprint:**

1. **Add Component:** `FollowerAgentComponent`
2. **Set Team Leader** (in BeginPlay or via level setup)
3. **Initialize RL Policy** (optional, can be done in C++)

---

## Performance Optimization

### For 4 Agents (Good Performance)

```
SyncCommandToBlackboard: 0.5s interval
QueryRLPolicyPeriodic:   1.0s interval
No special optimization needed
```

### For 8 Agents (Adjust Intervals)

```
SyncCommandToBlackboard: 1.0s interval
QueryRLPolicyPeriodic:   1.5-2.0s interval
bQueryOnlyWhenObservationChanged = true
```

### For 16+ Agents (Heavy Optimization)

```
SyncCommandToBlackboard: 1.5s interval
QueryRLPolicyPeriodic:   2.0-3.0s interval
bQueryOnlyWhenObservationChanged = true
ObservationSimilarityThreshold = 0.98 (stricter)
bRequireActiveCommand = true
Stagger query times using RandomDeviation
```

---

## Example: Minimal Working BT

**Required Blackboard Keys (Minimum):**
- `CommandType` (Enum: EStrategicCommandType, default: Idle)
- `IsCommandValid` (Bool, default: false)
- `TacticalAction` (Enum: ETacticalAction)

**Behavior Tree:**
```
Root
└─ Main Selector
   ├─ [Service: SyncCommandToBlackboard @ 0.5s]
   │  - CommandTypeKey: CommandType
   │  - IsCommandValidKey: IsCommandValid
   │
   ├─ [CheckCommandType: Assault, FlowAbort: LowerPriority]
   │  └─ Sequence (Offensive Subtree)
   │     ├─ [Service: QueryRLPolicyPeriodic @ 1.0s]
   │     │  - TacticalActionKey: TacticalAction
   │     └─ BTTask_ExecuteAssault
   │
   └─ [No Decorator - Fallback]
      └─ Wait Task (handles Idle state)
```

**That's it!** This minimal BT will:
1. Sync commands from team leader via service
2. Branch when command is "Assault" AND valid
3. Query RL policy for tactical action
4. Execute assault with selected tactic
5. Wait in Idle branch if no active command (initial state)

---

## Next Steps After BT Creation

1. **Assign BT to AI Controller**
   ```cpp
   RunBehaviorTree(BT_FollowerAgent);
   ```

2. **Set Up Team Structure**
   - Create TeamLeaderComponent
   - Register followers
   - Issue commands

3. **Test Incrementally**
   - Start with one command type (e.g., Assault)
   - Add one tactical action (e.g., AggressiveAssault)
   - Verify execution
   - Add more branches

4. **Enable Debug Logging**
   - Set `bLogSync = true`
   - Set `bLogQueries = true`
   - Watch console for decision flow

5. **Train RL Policy**
   - Let agents execute commands
   - Rewards accumulate automatically
   - Policy improves over time

---

## Conclusion

This guide provides a quick reference for creating Behavior Tree assets for the SBDAPM system. The key principles are:

✅ **Hierarchical Branching:** Commands → Tactical Actions → Execution
✅ **Service-Driven Updates:** Periodic sync of commands and RL queries
✅ **Reactive Execution:** Flow abort modes for dynamic behavior
✅ **Performance Aware:** Configurable intervals for scalability

For detailed implementation information, see:
- **WEEK_13_IMPLEMENTATION_SUMMARY.md** - Execution tasks
- **WEEK_14_IMPLEMENTATION_SUMMARY.md** - Decorators and services
- **CLAUDE.md** - Overall architecture

**Happy building!** 🎮🤖

---

**Guide by:** Claude Code Assistant
**Date:** 2025-11-02
**Version:** 1.0
