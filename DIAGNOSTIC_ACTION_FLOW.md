# Action Flow Diagnostic Guide

## Issue Summary

Schola actions are not being executed, and the system is falling back to rule-based behavior.

## Root Causes Identified

### Problem 1: Schola Actions Not Received ❌
**Symptom:** Seeing `[RL ACTION]` logs instead of `[SCHOLA ACTION]` logs
**Cause:** `bScholaActionReceived` flag is FALSE when StateTree checks it
**Location:** `STTask_ExecuteObjective.cpp:116`

**Why This Happens:**
- Schola gRPC server not running (UE not connected to Python)
- `TacticalActuator.TakeAction()` not being called by Schola plugin
- Flag synchronization issue between TacticalActuator and StateTree

### Problem 2: ONNX Model Not Loaded ❌
**Symptom:** Seeing `[RULE-BASED]` logs from `RLPolicyNetwork`
**Cause:** `bUseONNXModel = false` OR `ModelInstance.IsValid() = false`
**Location:** `RLPolicyNetwork.cpp:318`

**Why This Happens:**
- `bUseONNXModel` defaults to FALSE in constructor (`RLPolicyNetwork.cpp:17`)
- `LoadPolicy()` was never called OR failed to load ONNX file
- No trained PPO model exists yet

---

## Diagnostic Logs Added

### 1. TacticalActuator.cpp:90-95
```cpp
UE_LOG(LogTemp, Warning, TEXT("🎮 [SCHOLA ACTUATOR] '%s': Received action from Python → Move=(%.2f,%.2f) Speed=%.2f, Flag=TRUE"),
    *GetNameSafe(Owner),
    ParsedAction.MoveDirection.X, ParsedAction.MoveDirection.Y, ParsedAction.MoveSpeed);

UE_LOG(LogTemp, Warning, TEXT("    → SharedContext.bScholaActionReceived = %d (should be TRUE)"),
    SharedContext.bScholaActionReceived ? 1 : 0);
```

**What to Look For:**
- ✅ **If you see this log:** Schola is sending actions successfully
- ❌ **If you DON'T see this log:** Schola gRPC server not connected

### 2. STTask_ExecuteObjective.cpp:134
```cpp
UE_LOG(LogTemp, Display, TEXT("📊 [POLICY MODE] '%s': bScholaActionReceived=%d → Using local RL policy"),
    *GetNameSafe(Pawn), SharedContext.bScholaActionReceived ? 1 : 0);
```

**What to Look For:**
- If `bScholaActionReceived=0`: Schola action wasn't received this tick
- This confirms Priority 2 path (local RL policy) is active

### 3. RLPolicyNetwork.cpp:331-333 (ONNX Mode)
```cpp
UE_LOG(LogTemp, Display, TEXT("✅ [ONNX MODEL] Action: Move=(%.2f,%.2f) Speed=%.2f Look=(%.2f,%.2f) Fire=%d"),
    Action.MoveDirection.X, Action.MoveDirection.Y, Action.MoveSpeed,
    Action.LookDirection.X, Action.LookDirection.Y, Action.bFire);
```

**What to Look For:**
- ✅ **If you see this log:** ONNX model is loaded and working
- ❌ **If you DON'T see this log:** Model not loaded, using fallback

### 4. RLPolicyNetwork.cpp:341-342 (Fallback Mode)
```cpp
UE_LOG(LogTemp, Warning, TEXT("⚠️ [POLICY FALLBACK] bUseONNXModel=%d, ModelInstance.IsValid()=%d"),
    bUseONNXModel ? 1 : 0, ModelInstance.IsValid() ? 1 : 0);
```

**What to Look For:**
- `bUseONNXModel=0`: Model loading was never attempted OR failed
- `ModelInstance.IsValid()=0`: Model instance creation failed

---

## Expected Log Flow (Healthy System)

### Scenario A: Schola Training Mode (Real-Time PPO)
```
🎮 [SCHOLA ACTUATOR] 'BP_FollowerAgent_C_4': Received action from Python → Move=(0.50,0.80) Speed=0.60, Flag=TRUE
    → SharedContext.bScholaActionReceived = 1 (should be TRUE)
🔗 [SCHOLA ACTION] 'BP_FollowerAgent_C_4': Move=(0.50,0.80) Speed=0.60, Look=(0.20,0.90), Fire=0
[MOVE EXEC DIRECT] 'BP_FollowerAgent_C_4': AddMovementInput(0.70, 0.50, 0.00), Speed=360.0
```

**Status:** ✅ Working correctly (Python → Schola → TacticalActuator → StateTree)

### Scenario B: Inference Mode (Trained ONNX Model)
```
📊 [POLICY MODE] 'BP_FollowerAgent_C_4': bScholaActionReceived=0 → Using local RL policy
✅ [ONNX MODEL] Action: Move=(0.30,0.70) Speed=0.80 Look=(0.10,0.90) Fire=1
[MOVE EXEC AI] 'BP_FollowerAgent_C_4': MoveToLocation(1200.0, 800.0, 100.0), Speed=480.0
```

**Status:** ✅ Working correctly (ONNX model inference)

### Scenario C: Rule-Based Fallback (Current Issue)
```
📊 [POLICY MODE] 'BP_FollowerAgent_C_4': bScholaActionReceived=0 → Using local RL policy
⚠️ [POLICY FALLBACK] bUseONNXModel=0, ModelInstance.IsValid()=0
🔧 [RULE-BASED] Action: Move=(0.00,1.00) Speed=0.50 Look=(0.00,1.00) Fire=0 Crouch=0
```

**Status:** ❌ **This is what you're seeing** (No Schola, No ONNX → Rule-based)

---

## How to Fix

### Fix Option 1: Enable Schola Real-Time Training (Recommended)

**Prerequisites:**
1. Unreal Engine must be running
2. Schola plugin must be enabled in project settings
3. Python environment must be running and connected

**Steps:**

1. **Start UE with Schola enabled:**
   - Open project in UE Editor
   - Ensure Schola plugin is enabled (Edit → Plugins → Schola)
   - Play in Editor (PIE) or Launch standalone

2. **Run Python training script:**
   ```bash
   cd Source/GameAI_Project/Scripts
   python train_rllib.py
   ```

3. **Verify connection:**
   - Python should connect to `localhost:50051` (Schola gRPC server)
   - You should see `[SCHOLA ACTUATOR]` logs in UE Output Log
   - test_env.py should succeed without `DEADLINE_EXCEEDED` error

**Expected Result:**
- Actions flow: Python (RLlib) → Schola gRPC → TacticalActuator → StateTree
- Logs show `[SCHOLA ACTUATOR]` followed by `[SCHOLA ACTION]`

---

### Fix Option 2: Load Trained ONNX Model (Inference Mode)

**If you have a trained model (rl_policy_network.onnx):**

1. **Place ONNX file in project:**
   ```
   Content/AI/Models/rl_policy_network.onnx
   ```

2. **Load model in Blueprint (BP_FollowerAgent):**
   - Open `BP_FollowerAgent` Blueprint
   - Find `FollowerAgentComponent`
   - In Details panel, find `TacticalPolicy` reference
   - Create a `URLPolicyNetwork` object if not present
   - Call `LoadPolicy("Content/AI/Models/rl_policy_network.onnx")` on BeginPlay

3. **Verify in code (alternative):**
   ```cpp
   // In FollowerAgentComponent::BeginPlay()
   if (TacticalPolicy)
   {
       FString ModelPath = FPaths::ProjectContentDir() + TEXT("AI/Models/rl_policy_network.onnx");
       TacticalPolicy->LoadPolicy(ModelPath);
   }
   ```

**Expected Result:**
- Logs show `[ONNX MODEL]` instead of `[RULE-BASED]`
- Actions use trained neural network policy

---

### Fix Option 3: Accept Rule-Based Fallback (Testing Only)

**If you just want to test without training:**

The system will continue using rule-based actions (GetActionRuleBased). This is fine for:
- Testing movement/combat systems
- Debugging non-AI features
- Prototyping without ML dependency

**Behavior:**
- Agents will follow simple heuristics:
  - Low health (< 30%) → Retreat away from enemies
  - No cover + enemies visible → Seek cover perpendicular to enemy
  - Healthy (> 70%) → Cautious advance
  - Default → Patrol forward

**Not suitable for:**
- Real-time training experiments
- Performance benchmarking
- Final gameplay

---

## Quick Diagnostic Checklist

Run through these checks in order:

- [ ] **UE is running:** Launch project in PIE or standalone
- [ ] **Schola plugin enabled:** Edit → Plugins → Schola (check enabled)
- [ ] **Python environment ready:** `cd Scripts && python -m pip install -r requirements.txt`
- [ ] **test_env.py succeeds:** `python test_env.py` (should not timeout)
- [ ] **Logs show SCHOLA ACTUATOR:** Check UE Output Log for `🎮 [SCHOLA ACTUATOR]`
- [ ] **If not using Schola:** ONNX model loaded via `LoadPolicy()`

---

## Common Errors

### Error 1: `DEADLINE_EXCEEDED` in test_env.py
**Cause:** UE Schola gRPC server not running
**Fix:** Launch UE project first, THEN run Python script

### Error 2: `[POLICY FALLBACK] bUseONNXModel=0`
**Cause:** `LoadPolicy()` was never called
**Fix:** Call `LoadPolicy()` in Blueprint or C++ BeginPlay

### Error 3: `[POLICY FALLBACK] ModelInstance.IsValid()=0`
**Cause:** ONNX file not found or NNE runtime failed
**Fix:** Verify file path, check NNERuntimeORTCpu plugin enabled

### Error 4: Seeing `[RL ACTION]` but not `[SCHOLA ACTION]`
**Cause:** Schola not connected or `bScholaActionReceived` flag issue
**Fix:** Verify Schola plugin active, check TacticalActuator initialization

---

## Next Steps

1. Recompile the C++ code with the new diagnostic logging
2. Launch UE and run a test scenario
3. Check Output Log for the diagnostic messages above
4. Based on the logs, follow the appropriate Fix Option

**Need more help?**
- Check logs for specific error patterns listed above
- Verify Schola plugin installation: `Plugins/Schola-1.3.0/`
- Ensure Python gRPC dependencies installed: `grpcio`, `schola-gym`
