# StateTree Termination Fix - Setup Guide

## Problem Diagnosed

The StateTree was terminating immediately because:
1. **Missing Idle State**: When agents start without an objective, the ExecuteObjective task returns `Succeeded` (see STTask_ExecuteObjective.cpp:87-95)
2. **No Fallback State**: The StateTree had no fallback state to keep it running when waiting for objectives
3. **Result**: StateTree terminates immediately after BeginPlay

## Solution Implemented

Created two new components:

### 1. STTask_Idle
**Location**: `Source/GameAI_Project/Public/StateTree/Tasks/STTask_Idle.h`

**Purpose**: Keeps StateTree running when no objective is assigned
- Returns `Running` continuously to prevent StateTree termination
- Transitions to ExecuteObjective when objective is received
- Transitions to Dead when agent dies

### 2. STCondition_HasObjective
**Location**: `Source/GameAI_Project/Public/StateTree/Conditions/STCondition_HasObjective.h`

**Purpose**: Simple condition to check if an active objective exists
- Binds to `FollowerContext.CurrentObjective` and `FollowerContext.bHasActiveObjective`
- Can be inverted to check for NO objective

---

## StateTree Asset Configuration (Required!)

You MUST configure the StateTree asset in Unreal Editor with this structure:

### Target Structure
```
Root (Selector)
├─ [IsAlive == false] → DeadState
│  └─ Task: STTask_Dead
├─ [HasObjective == true] → ExecuteObjectiveState
│  └─ Task: STTask_ExecuteObjective
└─ IdleState (fallback)
   └─ Task: STTask_Idle
```

---

## Step-by-Step Configuration

### 1. Open StateTree Asset
- Navigate to: `Content/Game/Blueprints/AI/StateTree/ST_FollowerStateTree`
- Double-click to open StateTree Editor

### 2. Configure Root (Selector)
- Root should be a **Selector** (tries states in order until one succeeds)
- If it's a Sequence, change it to Selector

### 3. State 1: DeadState
**Condition**:
- Add condition: `STCondition_IsAlive`
- Set `bCheckIfDead = true` (invert the check)
- Bind `bIsAlive` to `FollowerContext.bIsAlive`

**Task**:
- Add task: `STTask_Dead`
- Bind `StateTreeComp` to context

### 4. State 2: ExecuteObjectiveState
**Condition**:
- Add condition: `STCondition_HasObjective`
- Set `bInvertCondition = false` (check for objective exists)
- Bind `CurrentObjective` to `FollowerContext.CurrentObjective`
- Bind `bHasActiveObjective` to `FollowerContext.bHasActiveObjective`

**Task**:
- Add task: `STTask_ExecuteObjective`
- Bind `StateTreeComp` to context
- Configure parameters (MovementSpeedMultiplier, RotationSpeed, AimTolerance)

### 5. State 3: IdleState (Fallback)
**Condition**: NONE (always enters if previous states don't match)

**Task**:
- Add task: `STTask_Idle`
- Bind `StateTreeComp` to context

---

## Execution Flow

### 1. Agent Spawns (No Objective)
```
Root evaluates:
├─ DeadState: bIsAlive=true → SKIP
├─ ExecuteObjectiveState: HasObjective=false → SKIP
└─ IdleState: Always true → ENTER
   └─ STTask_Idle returns Running → StateTree stays active
```

### 2. Objective Received
```
Root re-evaluates (Selector checks conditions every tick):
├─ DeadState: bIsAlive=true → SKIP
├─ ExecuteObjectiveState: HasObjective=true → ENTER
│  └─ STTask_ExecuteObjective returns Running → Execute actions
└─ IdleState: Not evaluated (higher priority state matched)
```

### 3. Objective Completed
```
ExecuteObjective task returns Succeeded
Root re-evaluates:
├─ DeadState: bIsAlive=true → SKIP
├─ ExecuteObjectiveState: HasObjective=false (cleared) → SKIP
└─ IdleState: Always true → ENTER (back to waiting)
```

### 4. Agent Dies
```
Root re-evaluates:
├─ DeadState: bIsAlive=false → ENTER
│  └─ STTask_Dead handles death animation/cleanup
```

---

## Build Instructions

1. **Rebuild C++ Code**:
   ```bash
   # Close Unreal Editor first
   cd C:\Users\Foryoucom\Documents\GitHub\4d\SBDAPM

   # Rebuild project
   "C:\Program Files\Epic Games\UE_5.6\Engine\Build\BatchFiles\Build.bat" ^
       GameAI_ProjectEditor Win64 Development ^
       -Project="GameAI_Project.uproject" -WaitMutex -NoHotReload
   ```

2. **Open Unreal Editor**:
   - Launch GameAI_Project.uproject
   - Wait for compilation to finish

3. **Configure StateTree Asset** (see steps above)

4. **Test**:
   - Play in Editor
   - Check Output Log for:
     - `⏸️ [IDLE] 'AgentName': ENTER - Waiting for objective`
     - StateTree should stay Running
     - When objective received: `✅ [IDLE EXIT] 'AgentName': Objective received`

---

## Verification Checklist

- [ ] STTask_Idle.h/cpp compiled successfully
- [ ] STCondition_HasObjective.h/cpp compiled successfully
- [ ] StateTree asset configured with correct structure (3 states)
- [ ] DeadState uses STCondition_IsAlive (inverted)
- [ ] ExecuteObjectiveState uses STCondition_HasObjective
- [ ] IdleState has no condition (fallback)
- [ ] All tasks have StateTreeComp bound correctly
- [ ] Agents print "⏸️ [IDLE]" when spawned without objective
- [ ] StateTree stays Running (not Succeeded/Failed)
- [ ] Transitions to ExecuteObjective when objective assigned

---

## Troubleshooting

### StateTree Still Terminates Immediately
- **Check**: IdleState is the LAST state in Root (fallback position)
- **Check**: IdleState has NO condition (should always match)
- **Check**: STTask_Idle is properly bound to StateTreeComp

### ExecuteObjective Never Entered
- **Check**: STCondition_HasObjective bindings are correct
- **Check**: FollowerComponent is assigning objectives properly
- **Check**: Context.bHasActiveObjective is set to true

### Compile Errors
- **Check**: Module dependencies in GameAI_Project.Build.cs
- **Required modules**: "Core", "CoreUObject", "Engine", "AIModule", "StateTreeModule", "GameplayTags"

---

## Files Modified/Created

### Created
- `Source/GameAI_Project/Public/StateTree/Tasks/STTask_Idle.h`
- `Source/GameAI_Project/Private/StateTree/Tasks/STTask_Idle.cpp`
- `Source/GameAI_Project/Public/StateTree/Conditions/STCondition_HasObjective.h`
- `Source/GameAI_Project/Private/StateTree/Conditions/STCondition_HasObjective.cpp`

### To Be Configured (In Editor)
- `Content/Game/Blueprints/AI/StateTree/ST_FollowerStateTree.uasset`

---

## Expected Log Output (Success)

```
🔵 UFollowerStateTreeComponent::BeginPlay CALLED for 'BP_Follower_C_1'
✅ StateTree successfully started and running!
⏸️ [IDLE] 'BP_Follower_C_1': ENTER - Waiting for objective (StateTree will keep running)
⏸️ [IDLE TICK] 'BP_Follower_C_1': Still idle - Objective=None, Alive=1
[2 seconds later]
⏸️ [IDLE TICK] 'BP_Follower_C_1': Still idle - Objective=None, Alive=1
[Objective assigned by leader]
✅ [IDLE EXIT] 'BP_Follower_C_1': Objective received (Eliminate), transitioning to execution
🎯 [EXEC OBJ] 'BP_Follower_C_1': ENTER - Objective: Eliminate, Health: 100.0%, Returning RUNNING
🔄 [EXEC OBJ TICK] 'BP_Follower_C_1': Tick #1 (DeltaTime=0.016), Alive=1, Objective=Eliminate
```

---

## Next Steps After Fix

1. **Verify Idle → Execute → Idle cycle works**
2. **Test objective assignment and completion**
3. **Test death transition (DeadState)**
4. **Test multiple agents simultaneously**
5. **Remove diagnostic logs** (marked with 🔍, ⏸️, 🎯 emojis) after verification

---

**Last Updated**: 2025-12-05
**Issue**: StateTree immediate termination
**Root Cause**: Missing Idle fallback state
**Status**: Awaiting Unreal Editor configuration
