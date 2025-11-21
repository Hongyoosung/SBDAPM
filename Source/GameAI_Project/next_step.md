# Next Steps: Agent Proximity & Formation Analysis

## Current Status Summary

**✅ WORKING:**
- MCTS team-level strategic planning (~34ms per decision)
- State Tree task execution (Assault, Defend, Support, Move, Retreat)
- Combat system (movement, firing, damage, rewards)
- Random actions at simulation start (exploration phase)
- Command pipeline from perception to execution

**⚠️ CURRENT ISSUE:**
During initial simulation, agents tend to be too close to each other. This raises the question:
- Is there task logic forcing agents to cluster?
- Or is this natural behavior during early learning/random exploration?

## State Tree Structure (Current - Working)
```
Root
├── Dead State (condition: !IsAlive)
├── Idle State (condition: CommandType == Idle)
├── Assault State (condition: CommandType == Assault)
│   ├── Task 1: STTask_QueryRLPolicy → Selects tactical action
│   └── Task 2: STTask_ExecuteAssault → Executes movement + firing
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

Evaluators: STEvaluator_SyncCommand, STEvaluator_UpdateObservation
Transitions: All states can transition to any other state based on command changes
```

## Agent Proximity Investigation Plan

### Phase 1: Agent Spacing Analysis (PRIORITY)
**Goal:** Understand why agents cluster together during initial simulation

**Hypotheses to Test:**
1. **MCTS assigns same/similar targets to all agents** → All agents converge on same location
2. **No formation spacing enforcement** → Agents ignore teammate positions
3. **Random actions lack spatial awareness** → Early exploration doesn't include spreading tactics
4. **Natural convergence** → Both teams moving toward objectives creates unavoidable clustering
5. **Task logic forces proximity** → ExecuteAssault/Move tasks have inherent clustering behavior

**Investigation Steps:**

1. **Log MCTS Command Assignments** (`Team/TeamLeaderComponent.cpp`)
   ```cpp
   // After MCTS decision, log each agent's target location
   for (const auto& Pair : Commands)
   {
       AActor* Agent = Pair.Key;
       FVector TargetLoc = Pair.Value.TargetLocation;
       UE_LOG(LogTemp, Warning, TEXT("[MCTS] Agent '%s': Command=%s, TargetLoc=%s"),
           *Agent->GetName(), *UEnum::GetValueAsString(Pair.Value.CommandType), *TargetLoc.ToString());
   }
   ```
   - Check if all agents get same target location
   - Verify MCTS generates diverse positioning

2. **Log Agent Positions and Inter-Agent Distances** (`FollowerAgentComponent.cpp` or `TeamLeaderComponent.cpp`)
   ```cpp
   // Log every 2 seconds
   for (AActor* Agent1 : TeamAgents)
   {
       for (AActor* Agent2 : TeamAgents)
       {
           if (Agent1 != Agent2)
           {
               float Distance = FVector::Dist(Agent1->GetActorLocation(), Agent2->GetActorLocation());
               UE_LOG(LogTemp, Log, TEXT("[FORMATION] Distance '%s' <-> '%s': %.1f cm"),
                   *Agent1->GetName(), *Agent2->GetName(), Distance);
           }
       }
   }
   ```
   - Identify if agents start apart and converge, or start close
   - Track minimum inter-agent distance over time

3. **Analyze Task Movement Logic** (Code review - no changes needed)
   - `STTask_ExecuteAssault.cpp:194` - `MoveToLocation(TargetLocation, 50.0f)`
   - `STTask_ExecuteMove.cpp` - Check destination calculation
   - Question: Do tasks consider ally positions when moving?
   - Current: Tasks only consider target/enemy positions, not teammates

4. **Check MCTS Reward Function** (`MCTS/MCTS.cpp:39-82`)
   - `FormationCoherence` exists in reward (line 46)
   - BUT: Does FormationCoherence encourage tight clustering or spacing?
   - Verify: What is the actual value of `TeamObs.FormationCoherence`?
   - Add logging: `UE_LOG(LogTemp, Warning, TEXT("[MCTS] FormationCoherence=%.2f"), TeamObs.FormationCoherence);`

**Expected Findings:**
- Likely: MCTS assigns similar target locations to all agents (e.g., all target same enemy)
- Likely: FormationCoherence may encourage clustering (high coherence = tight formation)
- Likely: No explicit spatial separation logic exists in task execution


## Implementation Order

### 1. LOGGING FOR PROXIMITY DIAGNOSIS (IMMEDIATE)
**Goal:** Gather data to understand clustering behavior

**Files to Modify:**
1. `Team/TeamLeaderComponent.cpp` - Log MCTS command assignments with target locations
2. `Team/TeamLeaderComponent.cpp` or `FollowerAgentComponent.cpp` - Log inter-agent distances
3. `MCTS/MCTS.cpp` - Log FormationCoherence value in CalculateTeamReward()

**Success Criteria:**
```
[MCTS] Agent 'BP_Follower_0': Command=Assault, TargetLoc=(1200, -300, 100)
[MCTS] Agent 'BP_Follower_1': Command=Assault, TargetLoc=(1180, -320, 100)  ← Similar location!
[MCTS] Agent 'BP_Follower_2': Command=Support, TargetLoc=(1150, -280, 100)  ← Also close!
[FORMATION] Distance 'BP_Follower_0' <-> 'BP_Follower_1': 180.0 cm  ← Too close!
[MCTS] FormationCoherence=0.85 (reward component: +42.5)
```

### 2. ROOT CAUSE ANALYSIS (AFTER LOGGING)
**Based on logs, identify the root cause:**

**Scenario A: MCTS assigns same target to all agents**
- **Solution:** Add target diversity to MCTS expansion
- Modify `TeamMCTSNode::Expand()` to ensure varied target assignments
- Add penalty for multiple agents targeting same location

**Scenario B: FormationCoherence encourages tight clustering**
- **Solution:** Redefine FormationCoherence calculation
- Change from "stay close together" to "maintain optimal spacing"
- Optimal spacing: 400-800cm between allies (close enough to support, far enough to avoid AoE)

**Scenario C: Task logic lacks spatial awareness**
- **Solution:** Add formation offset to task movement
- Modify `STTask_ExecuteAssault::ExecuteAggressiveAssault()` to add per-agent offset
- Example: Agent 0 → target directly, Agent 1 → target + 300cm left, Agent 2 → target + 300cm right

**Scenario D: Natural convergence (both teams meet)**
- **Solution:** Add initial positioning spread
- Modify spawn locations or initial commands to spread agents horizontally
- Add "Move" commands at start to position agents before engaging

### 3. FORMATION SPACING IMPLEMENTATION (AFTER ROOT CAUSE IDENTIFIED)
**Goal:** Implement solution based on root cause analysis

**Option 1: MCTS-Level Target Diversification**
```cpp
// In TeamMCTSNode.cpp or MCTS.cpp
// When generating command combinations, ensure target diversity:
TArray<FVector> UsedTargetLocations;
for (AActor* Follower : Followers)
{
    FVector BaseTarget = GetPrimaryEnemyLocation();
    FVector Offset = CalculateFormationOffset(Follower, UsedTargetLocations);
    Commands.Add(Follower, FStrategicCommand(Assault, BaseTarget + Offset));
    UsedTargetLocations.Add(BaseTarget + Offset);
}
```

**Option 2: Task-Level Formation Offset**
```cpp
// In STTask_ExecuteAssault.cpp
FVector TargetLocation = SharedContext.PrimaryTarget->GetActorLocation();
FVector FormationOffset = CalculateFormationOffset(Pawn, SharedContext.FollowerComponent);
FVector AdjustedTarget = TargetLocation + FormationOffset;
SharedContext.AIController->MoveToLocation(AdjustedTarget, 50.0f);
```

**Option 3: Observation-Based Learning (RL learns spacing)**
```cpp
// In ObservationElement.cpp - Add new feature
// "NearestAllyDistance" - Distance to nearest teammate
// RL policy will learn to maintain optimal spacing via rewards
Feature.NearestAllyDistance = CalculateNearestAllyDistance() / 3000.0f; // Normalize to 0-1
```


## Testing Checklist

### Phase 1: Proximity Diagnosis (Current Focus)
- [ ] Log MCTS command assignments (verify target location diversity)
- [ ] Log inter-agent distances every 2 seconds
- [ ] Log FormationCoherence values in MCTS reward calculation
- [ ] Identify if agents start spread and converge, or start clustered
- [ ] Determine root cause: MCTS assignments, task logic, or natural convergence

### Phase 2: Formation Spacing Implementation (After Diagnosis)
- [ ] Implement chosen solution (MCTS diversification, task offset, or RL observation)
- [ ] Verify agents maintain 400-800cm spacing from teammates
- [ ] Ensure formation doesn't break combat effectiveness
- [ ] Test with 2, 3, and 4 agent teams

### Phase 3: Tactical Behavior Validation
- [ ] MCTS generates diverse commands (Assault, Defend, Support, Retreat, Move)
- [ ] Agents spread out during initial positioning
- [ ] Agents maintain formation while advancing/retreating
- [ ] Flanking maneuvers work correctly (agents take different angles)
- [ ] Support agents position behind assault agents

### Integration Test (Already Working - Verify Maintained)
- [x] Full loop: Perception → MCTS → Command → State Transition → Movement → Combat → Damage → Rewards
- [x] Multiple agents coordinate (MCTS assigns different roles)
- [ ] Agents respond to dynamic threats with formation awareness
- [ ] Agents don't cluster when multiple targets available

## Files to Modify (Priority Order)

### Immediate: Proximity Diagnosis Logging
1. `Team/TeamLeaderComponent.cpp` - Log MCTS command assignments with target locations
2. `Team/TeamLeaderComponent.cpp` - Log inter-agent distances (or add to FollowerAgentComponent)
3. `MCTS/MCTS.cpp` - Log FormationCoherence value in CalculateTeamReward()

### After Diagnosis: Formation Spacing Implementation
4. `MCTS/MCTS.cpp` or `TeamMCTSNode.cpp` - Add target diversification (if root cause A)
5. `Observation/TeamObservation.h/cpp` - Redefine FormationCoherence calculation (if root cause B)
6. `StateTree/Tasks/STTask_ExecuteAssault.cpp` - Add formation offset (if root cause C)
7. `StateTree/Tasks/STTask_ExecuteMove.cpp` - Add formation offset (if root cause C)
8. Level Blueprint or spawn logic - Spread initial positions (if root cause D)

### Optional: Observation Enhancement
9. `Observation/ObservationElement.h/cpp` - Add NearestAllyDistance feature
10. `RL/RLPolicyNetwork.cpp` - Retrain with new observation feature

## Success Criteria

### Phase 1: Diagnosis Complete (Data Gathered)
```
[MCTS] Agent 'BP_Follower_0': Command=Assault, TargetLoc=(1200, -300, 100)
[MCTS] Agent 'BP_Follower_1': Command=Support, TargetLoc=(800, -280, 105)   ← Different target!
[MCTS] Agent 'BP_Follower_2': Command=Assault, TargetLoc=(1180, -500, 98)  ← Spread out!
[FORMATION] Distance 'BP_Follower_0' <-> 'BP_Follower_1': 520.0 cm  ← Good spacing!
[FORMATION] Distance 'BP_Follower_0' <-> 'BP_Follower_2': 450.0 cm  ← Good spacing!
[MCTS] FormationCoherence=0.75 (reward component: +37.5)
```

### Phase 2: Formation Spacing Working
```
[FORMATION] T=0s: Team spread = 0cm (just spawned)
[FORMATION] T=2s: Team spread = 450cm (moving to positions)
[FORMATION] T=5s: Team spread = 620cm (engaged, maintaining formation)
[FORMATION] T=10s: Team spread = 550cm (flanking, good spacing maintained)

[ASSAULT TASK] Agent 0: Moving to target with formation offset = (0, 0, 0)
[ASSAULT TASK] Agent 1: Moving to target with formation offset = (-300, 200, 0)  ← Offset!
[SUPPORT TASK] Agent 2: Moving to support position with formation offset = (0, -400, 0)  ← Behind!
```

### Phase 3: Full Tactical Behavior
```
[MCTS] Decision: 3 agents, 5 possible commands, diverse assignments
[MCTS] Agent 0: Assault (target: Enemy_1 at front)
[MCTS] Agent 1: Assault + FlankLeft (target: Enemy_1, offset left)
[MCTS] Agent 2: Support (target: cover position behind Agent 0)

[FORMATION] Inter-agent distances: 450-700cm (optimal range maintained)
[COMBAT] Agent 0 firing from front, Agent 1 firing from flank, Agent 2 suppressing
[COMBAT] Enemies flanked, no friendly clustering, good tactical spread
```
