# StateTree Diagnostic Checklist

## Symptom
- Evaluators only run TreeStart (not Tick)
- Task enters once, then exits immediately with "Exit (time: 0.0s)"
- No Tick logs from task or evaluators

## Root Cause
**StateTree asset bindings are missing or incorrect!**

## Fix Steps (UE5 Editor)

### 1. Open StateTree Asset
- Navigate to: `Content/.../FollowerStateTree` (or whatever your asset is named)
- Double-click to open in StateTree Editor

### 2. Check Evaluators Panel (Left Side)

#### STEvaluator_SyncObjective
- **CRITICAL**: Must have `StateTreeComp` input bound
- Click on the evaluator node
- In Details panel, find `StateTreeComp` property
- **Bind it to**: `FollowerStateTreeComponent` (from schema/context)
- If you don't see this binding option, **rebuild C++ first**!

#### STEvaluator_SpatialContext
- Same as above - bind `StateTreeComp` to `FollowerStateTreeComponent`

#### STEvaluator_UpdateObservation
- Should already be correct (no binding needed)

### 3. Check Root State Conditions

Look at the **ExecuteObjective** state (or root state):

#### STCondition_IsAlive
- `bIsAlive` input → Bind to `FollowerContext.bIsAlive`
- `bCheckIfDead` → Set to `false`

#### STCondition_CheckObjectiveType
- `CurrentObjective` → Bind to `FollowerContext.CurrentObjective`
- `bHasActiveObjective` → Bind to `FollowerContext.bHasActiveObjective`
- `AcceptedObjectiveTypes` → **MUST include `FormationMove`** (for Schola training!)
- `bRequireActiveObjective` → Set to `true`

### 4. Check State Transitions
- ExecuteObjective state should **NOT** have a transition that fires immediately
- Common mistake: A transition with conditions that are always true

### 5. Verify Task Bindings

Click on the **ExecuteObjective task** node:
- `StateTreeComp` → Bind to `FollowerStateTreeComponent`

## Expected Logs After Fix

```
[SYNC OBJECTIVE] TreeStart: Initialized SharedContext - Type=EObjectiveType::FormationMove
[SYNC OBJECTIVE] Tick #60: Type=EObjectiveType::FormationMove, Active=1  ← Should appear!
🎯 [EXEC OBJ] ENTER - Objective: EObjectiveType::FormationMove
🔄 [EXEC OBJ TICK] Tick #1 (DeltaTime=0.016)  ← Should appear!
🔄 [EXEC OBJ TICK] Tick #2 (DeltaTime=0.016)  ← Should continue!
[SCHOLA ACTION] Received action  ← Should appear when Python sends actions
```

## Common Mistakes

1. **Forgot to rebuild C++ before opening editor** → New properties won't show
2. **Bindings defaulted to "None"** → Must manually bind each input
3. **AcceptedObjectiveTypes doesn't include FormationMove** → Condition fails immediately
4. **StateTreeComp not bound** → Evaluators can't access shared context

## Debugging in Editor

1. **Enable StateTree Debugging**:
   - PIE (Play In Editor)
   - Window → Developer Tools → StateTree Debugger
   - Select your follower pawn
   - Watch which states/conditions are active

2. **Check Active States**:
   - Should show ExecuteObjective as RUNNING
   - If it shows SUCCEEDED or FAILED immediately, a condition failed

3. **Check Condition Values**:
   - StateTree Debugger shows condition evaluation results
   - Look for which condition returns FALSE
