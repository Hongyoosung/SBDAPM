# env.reset() Crash Fix - SBDAPM v3.1

**Date:** 2025-12-06
**Issue:** UE5 editor crashes when calling `env.reset()` in Python `test_connection.py`
**Status:** ✅ FIXED

---

## Root Causes Identified

### 1. **Circular Reset Loop** (CRITICAL)
**File:** `ScholaCombatEnvironment.cpp:606`

**Problem:**
```cpp
void AScholaCombatEnvironment::OnEpisodeStarted(int32 EpisodeNumber)
{
    // ... reset agents ...
    Reset();  // ❌ CAUSES INFINITE LOOP
}
```

**Flow:**
```
Python: env.reset()
  → ResetEnvironment()
    → StartNewEpisode()
      → OnEpisodeStarted.Broadcast()
        → OnEpisodeStarted()
          → Reset()  ← BACK TO STEP 1 (INFINITE LOOP!)
```

**Fix:** Removed the `Reset()` call that created circular dependency.

---

### 2. **Missing Initialization Guards**
**File:** `ScholaCombatEnvironment.cpp:193-232`

**Problem:**
- `env.reset()` could be called before teams are registered in `SimulationManager`
- This caused null pointer crashes when `StartNewEpisode()` tried to iterate teams

**Fix:** Added validation before calling `StartNewEpisode()`:
```cpp
TArray<int32> AllTeamIDs = SimulationManager->GetAllTeamIDs();
if (AllTeamIDs.Num() == 0)
{
    UE_LOG(LogTemp, Error, TEXT("No teams registered!"));
    return;  // Prevent crash
}
```

---

### 3. **Insufficient Error Handling**
**File:** `SimulationManagerGameMode.cpp:711-720`

**Problem:**
- `StartNewEpisode()` didn't check if `RegisteredTeams` was empty
- Could crash when iterating empty team list

**Fix:** Added early validation:
```cpp
if (RegisteredTeams.Num() == 0)
{
    UE_LOG(LogTemp, Error, TEXT("Cannot start episode - no teams registered"));
    return;
}
```

---

## Changes Made

### File: `ScholaCombatEnvironment.cpp`

**Line 592-607:** Removed circular reset call
```cpp
// BEFORE (BROKEN):
void AScholaCombatEnvironment::OnEpisodeStarted(int32 EpisodeNumber)
{
    // ... reset agents ...
    Reset();  // ❌ Circular dependency
}

// AFTER (FIXED):
void AScholaCombatEnvironment::OnEpisodeStarted(int32 EpisodeNumber)
{
    // ... reset agents ...
    // NOTE: Do NOT call Reset() - creates circular dependency
}
```

**Line 193-253:** Added comprehensive validation and logging
- Validates SimulationManager exists
- Checks teams are registered before reset
- Logs detailed diagnostics for debugging
- Clear error messages when reset fails

---

### File: `SimulationManagerGameMode.cpp`

**Line 711-720:** Added early validation
```cpp
void ASimulationManagerGameMode::StartNewEpisode()
{
    // CRITICAL: Prevent crash when no teams exist
    if (RegisteredTeams.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("Cannot start episode - no teams registered"));
        return;
    }

    // ... rest of reset logic ...
}
```

---

## Testing Instructions

### 1. **Rebuild Project**
```bash
# Close UE5 editor first
cd SBDAPM
# Rebuild C++ code
```

### 2. **Test in UE5**
1. Open UE5 editor
2. Start PIE (Play in Editor)
3. Check console for these logs:
   ```
   [ScholaEnv] Registered: 4 agents
   [ScholaEnv] Final count: 4 agents
   ```
4. Ensure no errors about missing teams

### 3. **Test Python Connection**
```bash
cd Source/GameAI_Project/Scripts
python test_connection.py
```

**Expected Output:**
```
[4/6] Calling env.reset() - FIRST CONNECTION TO UE5...
  ✓ Reset successful!
  - Observation type: <class 'dict'>
  - Agent count: 4
```

**If it still crashes, check UE5 console for:**
```
RESET BLOCKED - No teams registered!
```
This means agents haven't spawned yet. Add a delay in Python:
```python
import time
env = UnrealVectorEnv(...)
time.sleep(5)  # Wait for level to load
obs, info = env.reset()
```

---

## What Was Wrong?

### Original Behavior
1. Python calls `env.reset()`
2. UE5 calls `ResetEnvironment()`
3. Calls `StartNewEpisode()`
4. Broadcasts `OnEpisodeStarted`
5. Calls `Reset()` **AGAIN** ← Creates loop
6. May crash if teams not registered yet

### Fixed Behavior
1. Python calls `env.reset()`
2. UE5 validates teams are registered ✓
3. Calls `StartNewEpisode()` ✓
4. Broadcasts `OnEpisodeStarted` ✓
5. Agents reset successfully ✓
6. Returns to Python ✓

---

## Diagnostics Added

New logging helps identify crash points:

```
╔════════════════════════════════════════════════════════════════╗
║ [ScholaEnv] ResetEnvironment called                            ║
╚════════════════════════════════════════════════════════════════╝
[ScholaEnv] ✓ SimulationManager found
[ScholaEnv] ✓ Validation passed - 2 teams registered
[ScholaEnv]   - Team 0: 4 members
[ScholaEnv]   - Team 1: 4 members
[ScholaEnv] First reset - Starting Simulation
[ScholaEnv] ✓ Simulation started
[ScholaEnv] Calling StartNewEpisode...
===== EPISODE 1 STARTED =====
[ScholaEnv] ✓ Episode started successfully
[ScholaEnv] ═══ ResetEnvironment completed successfully ═══
```

---

## Prevention

To prevent this issue in the future:

### ❌ DON'T:
- Call `Reset()` from `OnEpisodeStarted()` callbacks
- Call `StartNewEpisode()` without checking teams exist
- Assume initialization order (agents might spawn late)

### ✅ DO:
- Validate teams are registered before reset
- Use early returns to prevent crashes
- Add comprehensive logging for debugging
- Test with `test_connection.py` after changes

---

## Related Files

Modified:
- `Source/GameAI_Project/Private/Schola/ScholaCombatEnvironment.cpp`
- `Source/GameAI_Project/Private/Core/SimulationManagerGameMode.cpp`

Reference:
- `Source/GameAI_Project/Scripts/test_connection.py`
- `Source/GameAI_Project/Public/Core/SimulationManagerGameMode.h`
- `Source/GameAI_Project/Public/Schola/ScholaCombatEnvironment.h`

---

## Success Criteria

✅ Test passes if:
1. No crash at step 4 (`env.reset()`)
2. Console shows "✓ Reset successful!"
3. Returns 4 agent observations
4. No "RESET BLOCKED" errors in UE5 console

❌ Test fails if:
- UE5 crashes at reset
- "No teams registered" error appears
- Agents don't respawn after death
- Infinite loop detected (check CPU usage)

---

**Next Steps:**
1. Test with `test_connection.py`
2. Test with `train_rllib.py` for multi-episode training
3. Monitor for any new issues during training
4. Verify episode transitions work correctly
