# State Tree Quick Reference

## File Structure

```
Source/GameAI_Project/
├── Public/StateTree/
│   ├── FollowerStateTreeSchema.h         # Shared context (71+ features)
│   ├── FollowerStateTreeComponent.h      # Component wrapper
│   ├── Tasks/
│   │   ├── STTask_QueryRLPolicy.h        # Query RL for tactical action
│   │   ├── STTask_ExecuteDefend.h        # Execute defensive tactics
│   │   ├── STTask_ExecuteAssault.h       # Execute assault tactics
│   │   └── STTask_ExecuteSupport.h       # Execute support tactics
│   ├── Evaluators/
│   │   ├── STEvaluator_UpdateObservation.h  # Update observation (71 features)
│   │   └── STEvaluator_SyncCommand.h        # Sync command from leader
│   └── Conditions/
│       ├── STCondition_CheckCommandType.h   # Check command type
│       ├── STCondition_CheckTacticalAction.h # Check tactical action
│       └── STCondition_IsAlive.h            # Check if alive
└── Private/StateTree/
    └── (implementations)
```

---

## Context Schema (FFollowerStateTreeContext)

**Core Components:**
- `FollowerComponent` - UFollowerAgentComponent*
- `AIController` - AAIController*
- `TeamLeader` - UTeamLeaderComponent*
- `TacticalPolicy` - URLPolicyNetwork*

**Command State:**
- `CurrentCommand` - FStrategicCommand
- `bIsCommandValid` - bool
- `TimeSinceCommand` - float

**Tactical State:**
- `CurrentTacticalAction` - ETacticalAction (16 actions)
- `TimeInTacticalAction` - float
- `ActionProgress` - float (0-1)

**Observation:**
- `CurrentObservation` - FObservationElement (71 features)
- `PreviousObservation` - FObservationElement

**Targets:**
- `VisibleEnemies` - TArray<AActor*>
- `PrimaryTarget` - AActor*
- `DistanceToPrimaryTarget` - float

**Cover:**
- `CurrentCover` - AActor*
- `bInCover` - bool
- `NearestCoverLocation` - FVector
- `DistanceToNearestCover` - float

**RL State:**
- `AccumulatedReward` - float
- `LastReward` - float
- `bUseRLPolicy` - bool

---

## Task Lifecycle

### QueryRLPolicy
```cpp
EnterState() → Query policy → Update context → Succeeded
```

### ExecuteDefend (Continuous)
```cpp
EnterState() → Initialize
Tick(DeltaTime) → Execute tactic → Calculate reward → Running
ExitState() → Cleanup
```

---

## State Tree Asset Structure

```
ST_FollowerAgent (State Tree Asset)
│
├── Schema: FFollowerStateTreeContext
│
├── Evaluators (Global):
│   ├── STEvaluator_UpdateObservation (UpdateInterval=0.1s)
│   └── STEvaluator_SyncCommand
│
└── States:
    ├── DeadState [STCondition_IsAlive: bCheckIfDead=true]
    │   └── (No tasks - just wait)
    │
    ├── AssaultState [STCondition_CheckCommandType: Assault]
    │   ├── STTask_QueryRLPolicy (EnterState only)
    │   └── STTask_ExecuteAssault (Tick continuously)
    │
    ├── DefendState [STCondition_CheckCommandType: Defend]
    │   ├── STTask_QueryRLPolicy
    │   └── STTask_ExecuteDefend
    │
    ├── SupportState [STCondition_CheckCommandType: Support]
    │   ├── STTask_QueryRLPolicy
    │   └── STTask_ExecuteSupport
    │
    └── IdleState (Default)
        └── STTask_QueryRLPolicy
```

---

## Code Examples

### Setup in C++ Character

```cpp
// Header
UPROPERTY(BlueprintReadWrite, Category = "AI")
UFollowerStateTreeComponent* StateTreeComponent;

// Constructor
StateTreeComponent = CreateDefaultSubobject<UFollowerStateTreeComponent>(TEXT("StateTree"));
```

### Setup in Blueprint

1. Add Component: `FollowerStateTreeComponent`
2. Set Properties:
   - State Tree Asset: `ST_FollowerAgent`
   - Follower Component: (auto-found)
   - Auto Start State Tree: true

### Query RL Policy

```cpp
// In STTask_QueryRLPolicy::EnterState()
TArray<float> ActionProbs = Context.TacticalPolicy->Forward(Context.CurrentObservation.ToFloatArray());
ETacticalAction BestAction = FindMaxIndex(ActionProbs);
Context.CurrentTacticalAction = BestAction;
```

### Provide Reward

```cpp
// In Execute task
float Reward = CalculateDefensiveReward(Context, DeltaTime);
Context.AccumulatedReward += Reward;
Context.FollowerComponent->ProvideReward(Reward, false);
```

---

## Debugging

### Enable Debug Drawing

**In State Tree asset:**
- `STEvaluator_UpdateObservation::bDrawDebugInfo = true`
- `STTask_QueryRLPolicy::bDrawDebugInfo = true`
- `STTask_ExecuteDefend::bDrawDebugInfo = true`

**In FollowerStateTreeComponent:**
- `Context.bDrawDebugInfo = true`

### Console Logs

```
LogTemp: STTask_QueryRLPolicy: Selected action 'DefensiveHold' for pawn 'AI_Soldier_1'
LogTemp: STEvaluator_SyncCommand: Command changed from 'None' to 'Defend'
LogTemp: STTask_ExecuteDefend: Starting defense at X=1000.0 Y=500.0 Z=0.0
```

---

## Common Issues

### "FollowerComponent not found"
- Check `bAutoFindFollowerComponent = true`
- Or manually set `FollowerComponent` reference

### "State Tree not starting"
- Check `bAutoStartStateTree = true`
- Verify `StateTreeAsset` is assigned
- Check schema matches `FFollowerStateTreeContext`

### "RL policy returning invalid actions"
- Check `TacticalPolicy->IsModelLoaded()`
- Verify observation size = 71 features
- Check action count = 16

### "State not transitioning"
- Check condition `AcceptedCommandTypes` list
- Verify `bRequireValidCommand` settings
- Check `Context.bIsCommandValid`

---

## Performance Tips

1. **Reduce observation update frequency** - Set `UpdateInterval = 0.2` (5Hz) if 10Hz is too expensive
2. **Limit raycast count** - 16 is good, 8 is faster
3. **Reduce enemy detection range** - 3000cm → 2000cm if feasible
4. **Batch RL queries** - Use `RLQueryInterval = 3.0s` to avoid querying every tick

---

## Conversion Table

| Old (FSM + BT) | New (State Tree) |
|----------------|------------------|
| `UStateMachine` | `UFollowerStateTreeComponent` |
| `UState` | State node in State Tree asset |
| `BTTaskNode` | `FStateTreeTaskBase` |
| `BTService` | `FStateTreeEvaluatorBase` |
| `BTDecorator` | `FStateTreeConditionBase` |
| Blackboard | `FFollowerStateTreeContext` |
| `ChangeState()` | Automatic (via conditions) |
| `GetCurrentState()` | `StateTreeComponent->GetCurrentStateName()` |
