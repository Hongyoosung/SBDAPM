# Schola Integration Debug Guide

## Issues Fixed

### Issue 1: Ghost Agents (6 detected instead of 4)
**Root Cause:** Python was iterating over ALL agents from Schola's action space (including CDO "ScholaAgentComponent" and extra ghost agents).

**Fix:** `sbdapm_env.py:687-702` - Now only sends actions for the 4 filtered valid agents in `self._agent_ids`.

### Issue 2: Wrong Action Shape (4, 8) instead of (8,)
**Root Cause:** Code was creating batched actions `(batch_size, 8)` instead of single actions `(8,)` per agent.

**Fix:** `sbdapm_env.py:687-702` - Removed batching logic, now creates simple dict `{agent_id: action_array(8)}`.

### Issue 3: No Movement in UE5
**Root Cause:** Actions weren't reaching StateTree due to wrong format/shape.

**Fix:** Combined fixes above + added extensive debug logging to trace action flow.

---

## Expected Output After Fixes

### Python Side (sbdapm_env.py)
```
[DEBUG] Formatted 4 actions for agents: ['ScholaAgentComponent_001', 'ScholaAgentComponent_002', 'ScholaAgentComponent_003', 'ScholaAgentComponent_004']
[DEBUG]   ScholaAgentComponent_001: shape (8,), non-zero: 5, sample: [0.12, -0.34, 0.56]
[DEBUG]   ScholaAgentComponent_002: shape (8,), non-zero: 6, sample: [-0.21, 0.45, 0.78]
[DEBUG]   ScholaAgentComponent_003: shape (8,), non-zero: 4, sample: [0.33, 0.11, 0.22]
[DEBUG]   ScholaAgentComponent_004: shape (8,), non-zero: 7, sample: [-0.44, 0.56, 0.89]
```

**Key Changes:**
- ✅ Only 4 agents (no CDO, no ghost agents)
- ✅ Shape is `(8,)` not `(4, 8)`
- ✅ Non-zero values confirm actions are being generated

### UE5 Side (TacticalActuator)
```
[TacticalActuator] 🎮 [SCHOLA ACTUATOR] 'BP_Follower_C_1': Received action from Python → Move=(0.12,-0.34) Speed=0.56, Look=(0.22,0.11), Fire=1
    → SharedContext.bScholaActionReceived = 1 (should be TRUE)
[TacticalActuator] 🎮 [SCHOLA ACTUATOR] 'BP_Follower_C_2': Received action from Python → Move=(-0.21,0.45) Speed=0.78, Look=(0.33,0.44), Fire=0
    → SharedContext.bScholaActionReceived = 1 (should be TRUE)
...
```

**Key Changes:**
- ✅ Non-zero move/look/speed values
- ✅ `bScholaActionReceived = 1` (TRUE)
- ✅ One log per valid agent (4 total)

### UE5 Side (STTask_ExecuteObjective)
```
[LogTemp] 🔗 [SCHOLA ACTION] 'BP_Follower_C_1': Move=(0.12,-0.34) Speed=0.56, Look=(0.22,0.11), Fire=1
[LogTemp] [MOVE EXEC DIRECT] 'BP_Follower_C_1': AddMovementInput(0.45, 0.33, 0.00), Speed=336.0
[LogTemp] [EXEC FIRE] ✅ 'BP_Follower_C_1': FIRING in direction (0.99, 0.10, 0.00)
```

**Key Changes:**
- ✅ "SCHOLA ACTION" log shows action is being used
- ✅ "MOVE EXEC DIRECT" shows movement is applied (NOT "MOVE EXEC STOP")
- ✅ Non-zero movement vectors and speeds

---

## Verification Checklist

### Before Running
1. ✅ Rebuild C++ code in Visual Studio/Rider
2. ✅ Close and restart UE5 Editor
3. ✅ Ensure 4 follower pawns have:
   - ScholaAgentComponent
   - FollowerAgentComponent
   - FollowerStateTreeComponent
4. ✅ Start PIE (Play In Editor) with Schola server enabled

### Run Python Script
```bash
cd C:\Users\PC\Documents\GitHub\SBDAPM\Source\GameAI_Project\Scripts
python train_rllib.py
```

### Monitor Logs

**Python Console:**
- Look for: `[DEBUG] Formatted 4 actions for agents`
- Verify: Only 4 agents listed (001-004)
- Verify: Shape is `(8,)` for each agent
- Verify: `non-zero` counts are > 0

**UE5 Output Log:**
- Filter for: `SCHOLA ACTUATOR`
- Verify: 4 logs with non-zero Move/Look/Speed
- Verify: `bScholaActionReceived = 1`

- Filter for: `SCHOLA ACTION`
- Verify: Actions are being read by StateTree

- Filter for: `MOVE EXEC`
- Verify: "MOVE EXEC DIRECT" (not "MOVE EXEC STOP")
- Verify: Non-zero AddMovementInput vectors

### Success Indicators
✅ **Python:** 4 agents, shape (8,), non-zero values
✅ **UE5 Actuator:** Actions received, bScholaActionReceived=TRUE
✅ **UE5 StateTree:** Actions applied, agents moving
✅ **UE5 Viewport:** Follower pawns visibly moving/aiming/firing

### Failure Indicators
❌ **Python:** More than 4 agents → Ghost agent filter failed
❌ **Python:** Shape (4, 8) → Batching logic regression
❌ **UE5:** "MOVE EXEC STOP" → Actions not reaching StateTree
❌ **UE5:** No "SCHOLA ACTUATOR" logs → gRPC connection issue

---

## Troubleshooting

### If still seeing 6 agents:
- Check `sbdapm_env.py:584-612` - filter logic should exclude CDO
- Verify Schola environment only has 4 ScholaAgentComponents in UE5

### If still seeing shape (4, 8):
- Ensure latest `sbdapm_env.py` is loaded (check file timestamp)
- Restart Python script completely

### If still seeing "MOVE EXEC STOP":
- Check TacticalActuator logs - are actions being received?
- Check if `bScholaActionReceived` is TRUE
- Verify action values are non-zero (movement magnitude > 0.01)

### If no "SCHOLA ACTUATOR" logs at all:
- gRPC connection not established
- Check Schola server is running in UE5 (PIE started)
- Check firewall/port 50051 is open
- Verify ScholaAgentComponents are registered with Schola environment

---

## Code Changes Summary

### Python Files
- `Scripts/sbdapm_env.py:687-702` - Simplified action formatting

### C++ Files
- `Private/Schola/TacticalActuator.cpp:90-97` - Enabled debug logging
- `Private/StateTree/Tasks/STTask_ExecuteObjective.cpp:123-127` - Enabled action debug
- `Private/StateTree/Tasks/STTask_ExecuteObjective.cpp:244-247` - Enabled movement debug

---

## Next Steps

1. **Rebuild C++ in Visual Studio**
2. **Restart UE5 Editor**
3. **Start PIE**
4. **Run `python train_rllib.py`**
5. **Monitor logs** (Python + UE5)
6. **Report back** with new log output

Expected: Agents should now move, aim, and fire based on RLlib actions!
