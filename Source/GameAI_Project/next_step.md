# Next Steps: Agent Execution Pipeline - COMPLETE DIAGNOSIS

## State Tree Structure (Current)
```
Root
├── Dead State (condition: !IsAlive)
│   └── (no tasks)
├── Idle State (condition: CommandType == Idle)
│   └── Task: Hold position (2s success delay)
├── Assault State (condition: CommandType == Assault)
│   ├── Task 1: STTask_QueryRLPolicy
│   └── Task 2: STTask_ExecuteAssault
├── Defend State (condition: CommandType == Defend/HoldPosition/TakeCover)
│   ├── Task 1: STTask_QueryRLPolicy
│   └── Task 2: STTask_ExecuteDefend
├── Support State (condition: CommandType == Support)
│   ├── Task 1: STTask_QueryRLPolicy
│   └── Task 2: STTask_ExecuteSupport
├── Retreat State (condition: CommandType == Retreat)
│   ├── Task 1: STTask_QueryRLPolicy
│   └── Task 2: STTask_ExecuteRetreat
└── Move State (condition: CommandType == MoveTo/Patrol/Advance)
    ├── Task 1: STTask_QueryRLPolicy
    └── Task 2: STTask_ExecuteMove

Evaluators (tick every frame):
- STEvaluator_SyncCommand (Priority: High) - Syncs CurrentCommand from FollowerAgentComponent
- STEvaluator_UpdateObservation (Priority: Medium) - Updates observation data

Transitions:
- Each state has transitions TO all other 6 states
- Transition conditions match target state's entry condition
- Example: Assault → Defend transition when CommandType changes to Defend
```

## Diagnostic Plan

### Phase 1: State Transition Diagnosis (PRIORITY)
**Goal:** Understand why agents stay in initial command state

1. **Add logging to STEvaluator_SyncCommand::Tick()**
   - Log `Context.CurrentCommand.Type` before and after sync
   - Log `FollowerAgent->GetCurrentCommand().Type`
   - Verify evaluator actually updates context

2. **Add logging to state transition conditions**
   - Log `STCondition_CheckCommandType::TestCondition()` results
   - Show: Expected vs Actual command type
   - Identify if conditions fail or never evaluated

3. **Verify TeamLeaderComponent command updates**
   - Log when leader issues NEW command to same follower
   - Check if `FollowerAgentComponent::CurrentCommand` actually updates
   - Confirm MCTS decisions vary over time

**Expected Findings:**
- Either: SyncCommand evaluator doesn't update context
- Or: Transition conditions never trigger
- Or: Leader never issues different commands to same agent

### Phase 2: Movement Diagnosis
**Goal:** Understand why MoveTo calls don't result in locomotion

1. **Add logging to STTask_ExecuteAssault::Tick()**
   - Log `AIController->MoveToLocation()` calls with target location
   - Log `AIController->GetMoveStatus()`
   - Log character velocity after MoveTo

2. **Verify AIController setup**
   - Check Blueprint: Does follower use AIController or PlayerController?
   - Verify: `bUseControllerRotationYaw` settings
   - Check: Character Movement Component exists and configured

3. **Verify NavMesh**
   - Check level has RecastNavMesh-Default
   - Press 'P' in editor to visualize NavMesh coverage
   - Verify target locations are on NavMesh

4. **Check task execution flow**
   - Log `EnterState()`, `Tick()`, `ExitState()` calls
   - Verify Tick() returns `EStateTreeRunStatus::Running` (not Succeeded/Failed prematurely)

**Expected Findings:**
- AIController is nullptr
- Or: No NavMesh in level
- Or: Task exits immediately (not running)
- Or: MoveTo called but pathfinding fails

### Phase 3: MCTS Command Variety
**Goal:** Understand why only Assault/Support appear

1. **Log MCTS rollout action selection**
   - File: `MCTS/MCTSNode.cpp` in `Rollout()` or action selection
   - Show: All available actions and their probabilities
   - Identify if other actions (Defend, Retreat) have zero probability

2. **Log observation features during MCTS**
   - Check team observation values during decision time
   - Verify: Enemy distance, health, ammo vary between simulations
   - Identify if observations are identical every run

3. **Check action space definition**
   - Verify `ECommandType` enum includes all 7+ types
   - Check MCTS maps to full action space (not hardcoded subset)

**Expected Findings:**
- MCTS rollout policy heavily biased toward Assault/Support
- Or: Observations don't vary (deterministic inputs → deterministic outputs)
- Or: Action space mapping incomplete


## Implementation Order

### 1. STATE TRANSITIONS (CRITICAL - BLOCKS ALL ELSE)
**Files:**
- `StateTree/Evaluators/STEvaluator_SyncCommand.cpp`
- `StateTree/Conditions/STCondition_CheckCommandType.cpp`
- `Team/FollowerAgentComponent.cpp`

**Actions:**
1. Add comprehensive logging (see Phase 1 above)
2. Run simulation, observe if:
   - Commands actually change in FollowerAgentComponent
   - SyncCommand evaluator updates context
   - Transition conditions trigger
3. Fix identified issue (likely: context not updating, or conditions not evaluating)

**Success Criteria:**
```
[SYNC CMD] Agent 0: Syncing command Assault → Defend
[TRANSITION] Agent 0: Assault State → Defend State (CommandType changed)
[STATE TREE] Agent 0: Entered Defend State
```

### 2. MOVEMENT EXECUTION
**Files:**
- `StateTree/Tasks/STTask_ExecuteAssault.cpp`
- `StateTree/Tasks/STTask_ExecuteMove.cpp`
- Blueprint: `BP_TestFollowerAgent`

**Actions:**
1. Add movement logging (see Phase 2 above)
2. Verify AIController setup in Blueprint
3. Check NavMesh coverage in level
4. Fix identified issue (likely: no AIController, or task flow broken)

**Success Criteria:**
```
[ASSAULT TASK] Agent 0: Moving to target at (1200, -300, 100)
[AI CONTROLLER] Agent 0: MoveStatus = InProgress, Velocity = (450, 0, 0)
[MOVEMENT] Agent 0: Distance to target = 850 units
```

### 3. MCTS COMMAND DIVERSITY
**Files:**
- `MCTS/MCTSNode.cpp` (rollout policy)
- `Team/TeamLeaderComponent.cpp` (observation building)

**Actions:**
1. Log action probabilities during rollout (see Phase 3 above)
2. Log observation variance between runs
3. Add randomness to rollout if deterministic
4. Verify observation features vary with game state

**Success Criteria:**
```
[MCTS] Rollout actions: Assault(0.35), Defend(0.25), Support(0.20), Retreat(0.10), Move(0.10)
[MCTS] Agent 0: Selected Defend (previously was Assault)
[MCTS] Agent 1: Selected Retreat (health low, enemies close)
```


## Testing Checklist

### Phase 1: State Transitions
- [ ] FollowerAgentComponent CurrentCommand updates when leader issues new commands
- [ ] STEvaluator_SyncCommand updates StateTree context every tick
- [ ] STCondition_CheckCommandType evaluates correctly for all command types
- [ ] State transitions occur when CurrentCommand.Type changes (e.g., Assault → Defend)
- [ ] Agents execute different commands over time (not stuck on initial command)

### Phase 2: Movement
- [ ] AIController exists and is assigned to follower actors
- [ ] NavMesh covers play area (visualize with 'P' key in editor)
- [ ] STTask_ExecuteAssault calls AIController->MoveToLocation()
- [ ] Character velocity changes after MoveTo call
- [ ] Agent physically moves toward target location
- [ ] Movement completes or continues until target reached

### Phase 3: Command Diversity
- [ ] MCTS generates Defend commands (not just Assault/Support)
- [ ] MCTS generates Retreat commands
- [ ] MCTS generates Move/Patrol commands
- [ ] Command assignments vary between simulation runs
- [ ] Same agent receives different commands across runs (not deterministic)
- [ ] Observation features vary between MCTS decisions

### Integration Test
- [ ] Full loop: Perception → MCTS → Command → State Transition → Movement → Combat → Damage → Rewards
- [ ] Multiple agents coordinate (one assaults, one supports)
- [ ] Agents respond to dynamic threats (switch from Assault to Retreat when health low)

## Files to Modify (Priority Order)

### Critical Path (Phase 1):
1. `StateTree/Evaluators/STEvaluator_SyncCommand.cpp` - Add logging, verify context update
2. `StateTree/Conditions/STCondition_CheckCommandType.cpp` - Add logging, verify evaluation
3. `Team/FollowerAgentComponent.cpp` - Log CurrentCommand changes
4. `Team/TeamLeaderComponent.cpp` - Log when issuing new commands to same follower

### Movement (Phase 2):
5. `StateTree/Tasks/STTask_ExecuteAssault.cpp` - Add movement/firing logs
6. `StateTree/Tasks/STTask_ExecuteMove.cpp` - Add movement logs
7. Blueprint: `BP_TestFollowerAgent` - Verify AIController setup

### Command Diversity (Phase 3):
8. `MCTS/MCTSNode.cpp` - Log rollout action selection, add randomness if needed
9. `Team/TeamLeaderComponent.cpp` - Log observation variance

## Success Criteria

### Minimal Success (Phase 1 + 2):
```
[TEAM LEADER] Issuing Assault to Agent 0, target: Enemy_1
[FOLLOWER] Agent 0: Command updated Idle → Assault
[SYNC CMD] Agent 0: Syncing command to StateTree context (Assault)
[TRANSITION] Agent 0: Idle State → Assault State
[ASSAULT TASK] Agent 0: EnterState, Target = Enemy_1 at (1200, -300, 100)
[ASSAULT TASK] Agent 0: Tick - Moving to target, distance = 1500
[AI CONTROLLER] Agent 0: MoveToLocation called, path found
[MOVEMENT] Agent 0: Velocity = (450, 120, 0), moving toward target

... 3 seconds later ...

[TEAM LEADER] Issuing Defend to Agent 0, target: CoverPoint_5
[FOLLOWER] Agent 0: Command updated Assault → Defend
[SYNC CMD] Agent 0: Syncing command to StateTree context (Defend)
[TRANSITION] Agent 0: Assault State → Defend State
[DEFEND TASK] Agent 0: EnterState, moving to cover at (800, 500, 100)
```

### Full Success (All Phases):
```
[MCTS] Rollout: Available actions = {Assault, Defend, Support, Retreat, Move}
[MCTS] Agent 0: Selected Assault (prob=0.40), Agent 1: Selected Support (prob=0.25)
[ASSAULT TASK] Agent 0: Firing at Enemy_1 (distance: 850, in range)
[SUPPORT TASK] Agent 1: Firing at Enemy_1 (supporting Agent 0)

... next MCTS decision (different commands) ...
[MCTS] Agent 0: Selected Retreat (health low), Agent 1: Selected Defend (hold position)
[TRANSITION] Agent 0: Assault State → Retreat State
[RETREAT TASK] Agent 0: Moving to safe location (900, 800, 100)
```
