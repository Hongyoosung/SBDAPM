# Next Steps: Agent Execution Pipeline

## Problem
Agents receive Assault commands but don't execute (stuck, no movement/firing).

**Logs show:**
- ✅ Commands received
- ✅ State transitions (Idle → Assault)
- ✅ target assigned
- ❌ No actual execution (movement/weapon firing)

## Root Causes
1. **ExecuteAssault Task** - Not moving agent or firing weapon, may not working state tree asset

## Implementation Order

### 1. Fix ExecuteAssault Task (`StateTree/Tasks/STTask_ExecuteAssault.cpp`)
**What:** Make agents move toward enemies and fire weapons
**Code locations:**
- `EnterState()` - Initialize assault behavior
- `Tick()` - Move toward target, fire when in range
- Integration: AIController MoveTo + WeaponComponent Fire

**Key logic:**
```cpp
// Get target from StateTree context
// AIController->MoveToLocation(TargetLocation)
// if (InWeaponRange) { WeaponComponent->Fire(Target) }
```

### 2. Target Assignment (`Team/TeamLeaderComponent.cpp`)
**What:** Assign nearest enemy when issuing Assault commands
**Code location:** `IssueCommandToFollower()` or MCTS action application

**Key logic:**
```cpp
// Get nearest enemy from SimulationManager
// Set FTeamCommand::TargetActor
// Pass to follower
```

### 3. StateTree Asset Validation
**What:** Ensure ST_FollowerBehavior has proper states/tasks
**Check in UE5 Editor:**
- Assault state exists
- STTask_ExecuteAssault linked to Assault state
- Transitions: Command change → State change
- Evaluators: SyncCommand runs first

### 4. Movement Integration
**What:** Ensure followers have AIController
**Files:** `FollowerAgentComponent.cpp`, Blueprint setup
**Requirements:**
- AAIController possession
- NavMesh in level
- MoveTo validation

### 5. Weapon Firing Validation
**What:** Confirm WeaponComponent fires correctly
**Already implemented:** `WeaponComponent::Fire()` exists
**Verify:** Projectile spawning, damage application

## Testing Checklist
- [ ] ExecuteAssault moves agent toward target
- [ ] WeaponComponent fires when in range
- [ ] Target assigned in Assault commands
- [ ] StateTree logs show task execution
- [ ] Damage applied to enemies
- [ ] RL rewards triggered on kill/damage

## Files to Modify
1. `StateTree/Tasks/STTask_ExecuteAssault.cpp` (PRIMARY)
2. `Team/TeamLeaderComponent.cpp` (target assignment)
3. `StateTree/FollowerStateTreeComponent.cpp` (task execution logging)
4. Blueprint: `BP_TestFollowerAgent` (AIController setup)

## Success Criteria
**Logs should show:**
```
[ASSAULT TASK] Agent X moving to enemy at (x,y,z)
[ASSAULT TASK] Agent X firing at enemy (distance: Y)
[WEAPON] Agent X dealt 10 damage to enemy
[RL REWARD] Agent X received +5.0 reward (damage dealt)
```
