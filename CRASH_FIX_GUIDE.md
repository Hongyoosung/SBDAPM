# UE5 PIE Crash Fix Guide

## Problem
UE5 crashes immediately after running PIE and connecting Python's `train_rllib.py`. No crash logs available.

## ✅ CRITICAL ROOT CAUSE IDENTIFIED

### **NULL POINTER CRASH in OnFollowerRespawned()**
   - **Location**: `FollowerStateTreeComponent.cpp:610`
   - **Code**: `GetWorld()->GetTimerManager().SetTimerForNextTick(...)`
   - **Issue**: `GetWorld()` returns NULL during first `env.reset()` call from Python

**Crash Sequence:**
1. Python calls `env.reset()` (FIRST connection)
2. `ResetEnvironment()` → `StartNewEpisode()` → `MarkAsAlive()` → `OnFollowerRespawned()`
3. `GetWorld()` returns NULL because components aren't fully initialized
4. NULL pointer dereference → **CRASH**

**Fix Applied**: Added World validity check before accessing TimerManager

## Other Potential Issues (Also Fixed)

### 1. **StateTree Not Ready During Observation Collection**
   - **Location**: `TacticalObserver::CollectObservations()`
   - **Issue**: Observer tries to access FollowerAgent before it's fully initialized
   - **Fix Applied**: Added `IsValidLowLevel()` and owner checks

### 2. **StateTree Not Ready During Action Execution**
   - **Location**: `TacticalActuator::TakeAction()`
   - **Issue**: Actuator tries to write to StateTree context when StateTree is Failed/Unset
   - **Fix Applied**: Check StateTree status before accessing context

### 3. **MarkAsAlive() Called Too Early**
   - **Location**: `FollowerAgentComponent::MarkAsAlive()`
   - **Issue**: Calls OnFollowerRespawned() when StateTree isn't ready
   - **Fix Applied**: Check StateTree validity and world before calling

## Fixes Applied

### C++ Changes (4 files modified):

1. **FollowerStateTreeComponent.cpp** (Lines 610-615) **← PRIMARY FIX**
   ```cpp
   // CRITICAL: Check World validity before accessing TimerManager
   UWorld* World = GetWorld();
   if (!World || !World->IsValidLowLevel())
   {
       UE_LOG(LogTemp, Warning, TEXT("⚠️ World not ready, skipping StateTree restart"));
       return;
   }
   ```

2. **FollowerAgentComponent.cpp** (Lines 398-405)
   ```cpp
   // Check StateTree validity before calling OnFollowerRespawned()
   if (StateTreeComp->IsValidLowLevel() && StateTreeComp->GetWorld())
   {
       StateTreeComp->OnFollowerRespawned();
   }
   ```

3. **TacticalObserver.cpp** (Line 68)
   ```cpp
   // Added IsValidLowLevel() check to prevent crash on invalid FollowerAgent
   if (!FollowerAgent || !FollowerAgent->IsValidLowLevel() || !FollowerAgent->GetOwner())
   ```

4. **TacticalActuator.cpp** (Lines 36-59)
   ```cpp
   // Added comprehensive validity checks:
   // - IsValidLowLevel() for FollowerAgent and StateTreeComp
   // - StateTree status check (Failed/Unset = skip action)
   ```

### Python Diagnostic Script:
- **test_connection.py** (NEW) - Isolates crash point step-by-step

## Testing Steps

### Step 1: Rebuild C++ Code

```bash
# Close UE5 Editor if open
# Open Visual Studio solution and rebuild

# OR use UE5 build tools:
UnrealBuildTool.exe GameAI_Project Win64 Development -waitmutex
```

### Step 2: Run Diagnostic Test

```bash
# 1. Start UE5 Editor
# 2. Click Play (PIE) - wait for level to load
# 3. In separate terminal:

cd Source/GameAI_Project/Scripts
python test_connection.py
```

**Expected Output:**
```
[1/6] Testing imports... ✓
[2/6] Creating connection... ✓
[3/6] Creating environment... ✓
[4/6] Calling env.reset()... ✓    ← **If crash occurs HERE**
[5/6] Taking single step... ✓     ← **Or HERE**
[6/6] Testing second reset... ✓
ALL TESTS PASSED
```

### Step 3: Interpret Results

#### ✅ **If test passes completely:**
```bash
# Great! Now try full training:
python train_rllib.py --iterations 5
```

#### ❌ **If crash at Step 4 (reset):**
**Problem**: Agent registration or StateTree initialization
**Check**:
- UE5 Output Log for last messages before crash
- Look for `[ScholaEnv]` or `[TacticalObserver]` errors
- Verify agents have FollowerStateTreeComponent attached

**Solutions**:
1. Check Blueprints have all required components:
   - UFollowerAgentComponent
   - UFollowerStateTreeComponent
   - UScholaAgentComponent
   - TacticalObserver/Actuator/RewardProvider

2. Verify StateTree asset is assigned in FollowerStateTreeComponent

3. Check for duplicate ScholaCombatEnvironment actors in level

#### ❌ **If crash at Step 5 (step):**
**Problem**: Action execution or StateTree tick
**Check**:
- Last log before crash should be `[SCHOLA ACTUATOR]` or `[StateTree]`
- StateTree might be in Failed state

**Solutions**:
1. Enable verbose StateTree logging:
   ```cpp
   // In FollowerStateTreeComponent.cpp, temporarily change line 102:
   UE_LOG(LogTemp, Warning, TEXT(...))  // Already verbose
   ```

2. Check ObjectiveManager is creating initial objectives properly

3. Verify all StateTree tasks have valid references

## Additional Diagnostics

### Enable Full Debug Logging

In `DefaultEngine.ini`:
```ini
[Core.Log]
LogTemp=VeryVerbose
LogStateTree=VeryVerbose
```

### Check StateTree Status Manually

In UE5 Console:
```
ShowDebug StateTree
```

### Monitor Python Connection

```bash
# Run with verbose Schola output:
python test_connection.py
# Watch UE5 Output Log simultaneously
```

## Common Issues & Solutions

### Issue: "StateTree not running after BeginPlay"
**Cause**: AIController not assigned or StateTree asset missing
**Fix**:
- Assign AIController in Blueprint
- Set StateTree asset in FollowerStateTreeComponent properties

### Issue: "FollowerAgent is NULL"
**Cause**: Components initialization order
**Fix**: Auto-find should work, but manually assign in editor if needed

### Issue: "CDO keys in action_space"
**Cause**: Schola detecting Class Default Object
**Fix**: Python wrapper already filters these - ignore warnings

### Issue: Multiple resets crash but first works
**Cause**: Episode reset not cleaning up properly
**Fix**: Check ObjectiveManager is clearing objectives on reset

## Verification Checklist

Before reporting issues, verify:

- [ ] Rebuilt C++ code after applying fixes
- [ ] PIE starts successfully without Python
- [ ] ScholaCombatEnvironment exists in level (only 1 instance)
- [ ] All follower agents have required components
- [ ] StateTree asset is assigned and valid
- [ ] Python can import Schola without errors
- [ ] Port 50051 is not blocked by firewall
- [ ] No other Schola servers running (check Task Manager)

## Rollback Instructions

If fixes cause new issues:

```bash
# Revert all changes:
git checkout Source/GameAI_Project/Private/Schola/TacticalObserver.cpp
git checkout Source/GameAI_Project/Private/Schola/TacticalActuator.cpp

# Rebuild
```

## Next Steps

Once diagnostic test passes:

1. **Short training run** (5 iterations):
   ```bash
   python train_rllib.py --iterations 5
   ```

2. **Monitor for crashes**:
   - Watch UE5 Output Log
   - Watch Python console
   - Check Windows Event Viewer if UE5 crashes

3. **Full training**:
   ```bash
   python train_rllib.py  # Default 100 iterations
   ```

## Files Modified

**C++ (4 files):**
- `Source/GameAI_Project/Private/StateTree/FollowerStateTreeComponent.cpp` **← PRIMARY FIX**
- `Source/GameAI_Project/Private/Team/FollowerAgentComponent.cpp`
- `Source/GameAI_Project/Private/Schola/TacticalObserver.cpp`
- `Source/GameAI_Project/Private/Schola/TacticalActuator.cpp`

**Python (1 file):**
- `Source/GameAI_Project/Scripts/test_connection.py` (NEW)

## Contact Points

If crash persists after these fixes, provide:
1. UE5 Output Log (last 100 lines before crash)
2. Python console output
3. Windows Event Viewer crash details (if available)
4. Which test step failed (1-6)
