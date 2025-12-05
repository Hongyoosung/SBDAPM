# StateTree Not Executing States - Issue Analysis

## Problem

StateTree starts successfully (`Status=Running`) but immediately stops without entering any task states.

## Evidence

### What We See:
```
✅ StateTree is now RUNNING
StateTree started immediately in BeginPlay!
✅ StateTree successfully started and running!
```

### What We DON'T See:
```
⏸️ [IDLE] ENTER           ← Idle task should enter
⏸️ [IDLE TICK]            ← Idle task should tick
🎯 [EXEC OBJ] ENTER       ← ExecuteObjective task should enter
```

## Root Cause

The **Blueprint StateTree asset** (`ST_FollowerStateTree`) is likely:

1. **Empty** - No states configured
2. **Broken Bindings** - Context data not bound to task inputs
3. **All Conditions Failing** - No states can be entered due to failed entry conditions
4. **Root State Immediately Succeeding** - Misconfigured root selector

## Diagnostic Steps

### 1. Check StateTree Asset in Editor

**Location:** `Content/Game/Blueprints/AI/StateTree/ST_FollowerStateTree.uasset`

**Open in Editor:**
1. Double-click the asset in Content Browser
2. Verify StateTree structure:

#### Expected Structure:
```
Root (Selector)
├─ State: Dead
│  ├─ Enter Conditions: [IsAlive == false]
│  └─ Task: STTask_Dead
│
├─ State: ExecuteObjective
│  ├─ Enter Conditions: [bHasActiveObjective == true]
│  └─ Task: STTask_ExecuteObjective
│      ├─ Bind: StateTreeComp → FollowerStateTreeComponent
│
└─ State: Idle (Fallback)
   ├─ Enter Conditions: None (fallback)
   └─ Task: STTask_Idle
      ├─ Bind: StateTreeComp → FollowerStateTreeComponent
```

### 2. Verify Task Bindings

**Critical Bindings for STTask_ExecuteObjective:**

| Parameter | Bind To | Source |
|-----------|---------|--------|
| `StateTreeComp` | FollowerStateTreeComponent | External Data |

**Critical Bindings for STTask_Idle:**

| Parameter | Bind To | Source |
|-----------|---------|--------|
| `StateTreeComp` | FollowerStateTreeComponent | External Data |

### 3. Verify Condition Bindings

**STCondition_IsAlive:**

| Parameter | Bind To | Source |
|-----------|---------|--------|
| `bIsAlive` | FollowerContext.bIsAlive | External Data |

**STCondition_CheckObjectiveType (if used):**

| Parameter | Bind To | Source |
|-----------|---------|--------|
| `CurrentObjective` | FollowerContext.CurrentObjective | External Data |
| `bHasActiveObjective` | FollowerContext.bHasActiveObjective | External Data |

### 4. Check Schema Assignment

**StateTree Asset → Details Panel:**
- Schema: `UFollowerStateTreeSchema` ✅
- If it shows `Default__StateTreeComponentSchema`, the asset is using the wrong schema!

### 5. Verify External Data Registration

In StateTree editor, check "External Data" panel:

**Required:**
- ✅ Pawn (APawn)
- ✅ Actor (AActor)
- ✅ FollowerContext (FFollowerStateTreeContext)
- ✅ FollowerComponent (UFollowerAgentComponent)
- ✅ FollowerStateTreeComponent (UFollowerStateTreeComponent)

**Optional:**
- AIController (AAIController) - can be NULL for Schola
- TeamLeader (UTeamLeaderComponent)
- TacticalPolicy (URLPolicyNetwork)

## Common Fixes

### Fix 1: Recreate StateTree Asset

If asset is corrupted:

1. **Backup existing asset:** Duplicate `ST_FollowerStateTree`
2. **Create new StateTree:**
   - Right-click in Content Browser
   - Gameplay → State Tree
   - Name: `ST_FollowerStateTree_New`
   - Schema: `FollowerStateTreeSchema`

3. **Add Root Selector:**
   - Root node should be "Selector" type

4. **Add States (in order):**

   **A. Dead State:**
   ```
   State Name: Dead
   Enter Conditions:
     - Add "Is Alive"
       - bCheckIfDead: true
       - Bind bIsAlive → FollowerContext.bIsAlive
   Tasks:
     - Add "STTask_Dead"
       - Bind StateTreeComp → FollowerStateTreeComponent
   ```

   **B. ExecuteObjective State:**
   ```
   State Name: ExecuteObjective
   Enter Conditions:
     - Add "Has Objective" (custom condition or manual bool check)
       - Bind bHasActiveObjective → FollowerContext.bHasActiveObjective
   Tasks:
     - Add "STTask_ExecuteObjective"
       - Bind StateTreeComp → FollowerStateTreeComponent
       - MovementSpeedMultiplier: 1.0
       - RotationSpeed: 360.0
       - AimTolerance: 5.0
   ```

   **C. Idle State (Fallback):**
   ```
   State Name: Idle
   Enter Conditions: None (this is the fallback)
   Tasks:
     - Add "STTask_Idle"
       - Bind StateTreeComp → FollowerStateTreeComponent
   ```

5. **Compile & Save**

6. **Reassign to BP_FollowerAgent:**
   - Open `BP_FollowerAgent`
   - Select `FollowerStateTreeComponent`
   - StateTree: Set to `ST_FollowerStateTree_New`
   - Compile & Save

### Fix 2: Verify External Data Bindings

**If bindings show "Invalid" or "Not Found":**

1. Open StateTree asset
2. Click "Compile" button (top toolbar)
3. Check Output Log for binding errors
4. Re-bind any invalid bindings manually

### Fix 3: Check for Empty Root

**Symptom:** Root node has no child states

**Fix:**
1. Right-click Root
2. Add State (Selector type if Root is Selector)
3. Add at minimum: Idle state with STTask_Idle

### Fix 4: Enable StateTree Debug Visualization

**In-Game Debugging:**

1. Console command: `showdebug statetree`
2. Look for agent's StateTree status overlay
3. Should show: Current state name, active tasks, condition results

**Editor Debugging:**

1. Play in Editor (PIE)
2. Select BP_FollowerAgent in World Outliner
3. Details panel → FollowerStateTreeComponent
4. Check "Run Status" and "Active States"

## Expected Behavior After Fix

### Logs on Startup (No Objective):
```
UFollowerStateTreeComponent:🔍 CheckRequirementsAndStart() CALLED for 'BP_FollowerAgent_C_8'
🔍 DIAGNOSTIC: StateTree asset info:
  → Asset Name: ST_FollowerStateTree
  → Schema: FollowerStateTreeSchema
  → Valid: 1
UFollowerStateTreeComponent:✅ FollowerComponent found
UFollowerStateTreeComponent:🚀 All requirements met! Calling StartLogic()...
🔵 UFollowerStateTreeComponent::SetContextRequirements SUCCESS
⏸️ [IDLE] 'BP_FollowerAgent_C_8': ENTER - Waiting for objective (StateTree will keep running)
UFollowerStateTreeComponent: ✅ StateTree is now RUNNING
UFollowerStateTreeComponent: ✅ StateTree successfully started and running!
⏸️ [IDLE TICK] 'BP_FollowerAgent_C_8': Still idle - Objective=None, Alive=1
```

### Logs on Objective Received:
```
⏸️ [IDLE EXIT] 'BP_FollowerAgent_C_8': Objective received (Eliminate), transitioning to execution
🎯 [EXEC OBJ] 'BP_FollowerAgent_C_8': ENTER - Objective: Eliminate, Health: 100.0%, Returning RUNNING
🔄 [EXEC OBJ TICK] 'BP_FollowerAgent_C_8': Tick #1 (DeltaTime=0.016), Alive=1, Objective=Eliminate, ScholaAction=0
```

## Next Steps

1. **Run with new diagnostic logging** - Compile C++ changes and check logs for "DIAGNOSTIC" messages
2. **Inspect StateTree asset in editor** - Verify structure matches expected layout
3. **Check bindings** - Ensure all task parameters are bound to correct external data
4. **Test in PIE** - Use `showdebug statetree` to visualize state execution
5. **Report findings** - Share diagnostic output and screenshots of StateTree editor

## Files to Check

- **StateTree Asset:** `Content/Game/Blueprints/AI/StateTree/ST_FollowerStateTree.uasset`
- **Character BP:** `Content/Game/Blueprints/Characters/BP_FollowerAgent.uasset`
- **Component Setup:** Verify FollowerStateTreeComponent has StateTree asset assigned

---

**Last Updated:** 2025-12-05 (Post Schola compatibility fix)
