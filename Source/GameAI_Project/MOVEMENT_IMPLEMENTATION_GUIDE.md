# SBDAPM Enhanced Movement System - Implementation Guide

## Overview

This guide explains the new **tactical movement system** that has been integrated into SBDAPM, replacing the simple directional movement actions with realistic 3D game movement behaviors.

**Last Updated:** 2025-10-27
**Version:** 2.0
**Related:** See `CLAUDE.md` for overall architecture

---

## What Has Changed?

### Old Movement System (Removed)
- **Simple Directional Actions:** MoveForward, MoveBackward, MoveLeft, MoveRight
- **Behavior:** Agent moved in cardinal directions (up/down/left/right)
- **Use Case:** Very basic movement, not realistic for 3D games

### New Movement System (Current)
- **Tactical Movement Actions:** MoveToAlly, MoveToAttackPosition, MoveToCover, TacticalReposition
- **Behavior:** Agent makes strategic decisions based on the 71-feature observation space
- **Use Case:** Realistic combat movement (flanking, cover-seeking, repositioning, ally coordination)

---

## New Movement Actions

### 1. MoveToAllyAction

**File:** `Actions/Movement/MoveToAllyAction.h/cpp`

**Purpose:** Move toward friendly units for coordination and support.

**Strategic Decisions:**
- Calculate centroid of nearby allies
- Move to maintain formation while avoiding clustering
- Prioritize when low health or facing many enemies

**Key Parameters:**
- `MinAllyDistance = 300.0f` - Minimum spacing from allies
- `MaxAllySearchRadius = 2000.0f` - How far to search for allies

**When MCTS Selects This:**
- Low health (< 50%) → seek ally support
- Many enemies (≥ 3) → safety in numbers
- Low stamina → regroup and recover

**Implementation Requirements:**
- Allies must be tagged with `"Ally"` or `"Friendly"` tag in Unreal
- Uses `UGameplayStatics::GetAllActorsOfClass()` to find allies
- Sets destination to centroid of ally positions

---

### 2. MoveToAttackPositionAction

**File:** `Actions/Movement/MoveToAttackPositionAction.h/cpp`

**Purpose:** Find and move to optimal attack position based on tactical strategy.

**Attack Strategies:**
- **Flanking:** Move to enemy's side/rear (90°+ angle)
- **High Ground:** Seek elevated positions
- **Cover-Based:** Position near cover with line of sight
- **Direct Assault:** Move straight toward enemy at optimal range
- **Encircle:** Surround enemy (multi-agent)

**Key Parameters:**
- `OptimalAttackRange = 800.0f` - Preferred distance from enemy
- `MinAttackRange = 300.0f` - Minimum safe distance
- `FlankingAngleMin = 90.0f` - Minimum angle for flanking
- `HighGroundHeightBonus = 200.0f` - Z-height advantage to seek

**When MCTS Selects This:**
- MoveToState decides agent should engage enemy
- Agent has good health/stamina
- Target enemy identified

**Strategy Selection Logic:**
```cpp
// Low health → cover-based
if (Health < 40% && HasCover) → CoverBased

// Multiple enemies → flank
if (Enemies >= 2) → Flanking

// Enemy too close → create distance via flanking
if (EnemyDistance < MinAttackRange) → Flanking

// High health/stamina → take high ground
if (Health > 80% && Stamina > 70%) → HighGround

// Default → direct assault
else → DirectAssault
```

---

### 3. MoveToCoverAction

**File:** `Actions/Movement/MoveToCoverAction.h/cpp`

**Purpose:** Move to the nearest safe cover position.

**Cover Detection Methods:**
1. **Primary:** Use `FObservationElement.bHasCover`, `NearestCoverDistance`, `CoverDirection`
2. **Fallback:** Analyze raycast data for `ERaycastHitType::Cover`

**Key Parameters:**
- `MaxCoverSearchRadius = 1500.0f` - Maximum distance to search
- `MinCoverDistance = 100.0f` - Don't move if already very close
- `PreferredCoverDistance = 500.0f` - Ideal distance to cover

**Cover Urgency Calculation:**
```cpp
Urgency += (Health < 30%) ? 0.5 : 0.0
Urgency += (Health < 60%) ? 0.3 : 0.0
Urgency += (Enemies >= 3) ? 0.3 : 0.0
Urgency += (NearestEnemy < 300) ? 0.3 : 0.0
Urgency += (Shield == 0 && Health < 80%) ? 0.2 : 0.0
Urgency += (WeaponCooldown > 2s) ? 0.1 : 0.0
```

**Cover Compromise Detection:**
- Enemy within 200 units → cover compromised
- Enemy flanking (angle > 90°) within 500 units → compromised
- Surrounded by 4+ enemies → compromised

**When MCTS Selects This:**
- Low health (< 40%)
- Under fire from multiple enemies
- Need to reload/heal safely

---

### 4. TacticalRepositionAction

**File:** `Actions/Movement/TacticalRepositionAction.h/cpp`

**Purpose:** Dynamically reposition based on changing combat situation.

**Position Evaluation Factors:**
```cpp
Quality = 0.5 (base)

// Cover (weight: 0.4)
+= HasCover ? (1.0 - CoverDistance/500) * 0.4 : -0.2

// Enemy Proximity (weight: 0.3)
-= (EnemyDistance < 300) ? 0.3 : 0.0
-= (EnemyDistance < 600) ? 0.15 : 0.0
-= (FlankingEnemies >= 2) ? 0.3 : 0.0

// Visibility (weight: 0.1)
+= AverageRaycastDistance * 0.1

// Health Status
-= (Health < 50% && !HasCover) ? 0.2 : 0.0

// Surrounded
-= (Enemies >= 3) ? 0.25 : 0.0
```

**Reposition Triggers:**
- Position quality < `RepositionThreshold (0.4)`
- Health < 30% without cover
- Flanked by 2+ enemies within 400 units

**Key Parameters:**
- `RepositionThreshold = 0.4f` - Position quality threshold
- `MinRepositionDistance = 300.0f` - Minimum move distance
- `MaxRepositionDistance = 1000.0f` - Maximum move distance
- `CoverWeight = 0.4f` - Cover importance weight
- `AllyProximityWeight = 0.2f` - Ally proximity weight
- `EnemyAvoidanceWeight = 0.3f` - Enemy avoidance weight
- `VisibilityWeight = 0.1f` - Visibility weight

**When MCTS Selects This:**
- Current position is poor (quality < 0.4)
- Being flanked or surrounded
- Cover compromised

---

## Architecture Integration

### Strategic Layer (FSM + MCTS)

**MoveToState** manages high-level movement strategy:

```cpp
void UMoveToState::UpdateState(...)
{
    // MCTS evaluates all movement actions
    MCTS->RunMCTS(PossibleActions, StateMachine);

    // Determine movement mode
    FString MovementMode = "Normal";
    if (Health < 40% || Enemies >= 3)
        MovementMode = "Defensive";
    else if (Health > 80% && Stamina > 70%)
        MovementMode = "Aggressive";
    else if (Enemies > 0 && HasCover)
        MovementMode = "Tactical";

    // Could set Blackboard: "MovementMode"
}
```

**Available Actions:**
```cpp
TArray<UAction*> UMoveToState::GetPossibleActions()
{
    // NEW: Tactical actions
    Actions.Add(NewObject<UMoveToAllyAction>());
    Actions.Add(NewObject<UMoveToAttackPositionAction>());
    Actions.Add(NewObject<UMoveToCoverAction>());
    Actions.Add(NewObject<UTacticalRepositionAction>());

    // LEGACY: Directional actions (optional)
    Actions.Add(NewObject<UMoveForwardAction>());
    // ... etc
}
```

### Tactical Layer (Behavior Tree)

**Behavior Tree Integration:**

```
Root (Selector)
├─ [Decorator: Strategy == "MoveTo"] MoveToSubtree
│  ├─ [Service: UpdateObservation] Sequence
│  │  ├─ Task: BTTask_ExecuteMCTSDecision
│  │  ├─ Task: MoveTo (Blackboard: Destination)
│  │  ├─ Task: BTTask_UpdateProgress
│  │  └─ Task: BTTask_CheckArrival
```

**Required Blackboard Keys:**
- `CurrentStrategy` (String) - Set by MoveToState ("MoveTo", "Attack", "Flee")
- `Destination` (Vector) - Set by movement actions
- `MovementMode` (String) - Set by MoveToState ("Normal", "Defensive", "Aggressive", "Tactical")

**Required BT Tasks (to implement):**

#### BTTask_ExecuteMCTSDecision
```cpp
class UBTTask_ExecuteMCTSDecision : public UBTTaskNode
{
    // Reads: StateMachine component
    // Calls: MCTS to select action
    // Updates: Blackboard Destination
};
```

#### BTTask_UpdateProgress
```cpp
class UBTTask_UpdateProgress : public UBTTaskNode
{
    // Monitors: Distance to destination
    // Updates: Progress percentage
    // Returns: Success when close enough
};
```

---

## Observation System Requirements

**The new movement actions rely on the 71-feature observation system.**

### Critical Observation Fields

**Enemy Information:**
```cpp
int32 VisibleEnemyCount;                    // Total enemies
TArray<FEnemyObservation> NearbyEnemies;    // Top 5 enemies
    - Distance
    - Health
    - RelativeAngle
```

**Cover Information:**
```cpp
bool bHasCover;                  // Cover available?
float NearestCoverDistance;      // Distance to cover
FVector2D CoverDirection;        // Direction (normalized)
```

**Raycasts (16 rays at 22.5° intervals):**
```cpp
TArray<float> RaycastDistances;          // 16 distances (normalized)
TArray<ERaycastHitType> RaycastHitTypes; // Object types
    - None, Wall, Enemy, Cover, HealthPack, Weapon, Other
```

**Agent State:**
```cpp
FVector Position;       // World position
FVector Velocity;       // Current velocity
FRotator Rotation;      // Current rotation
float Health;           // 0-100
float Stamina;          // 0-100
float Shield;           // 0-100
```

### Updating Observations from Blueprint

**In your BT Service or Character Blueprint:**

```cpp
// Update enemy observations
TArray<FEnemyObservation> Enemies;
// ... populate enemies from perception system ...
StateMachine->UpdateEnemyObservations(Enemies);

// Update cover information
bool bHasCover = CheckCoverAvailable();
float CoverDist = GetNearestCoverDistance();
FVector2D CoverDir = GetCoverDirection();
StateMachine->UpdateCoverObservation(bHasCover, CoverDist, CoverDir);

// Update raycasts
TArray<float> Distances;
TArray<ERaycastHitType> HitTypes;
PerformRadialRaycast(Distances, HitTypes);
StateMachine->UpdateRaycastObservation(Distances, HitTypes);
```

---

## Blueprint Integration

### Tagging Allies

For `MoveToAllyAction` to work:

1. **In Character Blueprint:**
   - Select your ally characters
   - In Details panel → Actor → Tags
   - Add tag: `Ally` or `Friendly`

### Implementing Cover Detection

**Method 1: EQS (Environment Query System)**

```
// Create EQS_FindCover query
Context: Querier
Tests:
  - Distance (prefer closer)
  - Trace (line of sight to enemy)
  - Cover Quality (raycast from enemy)

// In Blueprint
RunEQSQuery(EQS_FindCover)
  → Success: Update StateMachine->bHasCover, CoverDirection
```

**Method 2: Manual Raycast**

```cpp
// In Blueprint or C++
FVector Start = GetActorLocation();
TArray<FHitResult> HitResults;

// Raycast in circle around agent
for (int i = 0; i < 16; ++i)
{
    float Angle = i * 22.5f;
    FVector Direction = GetActorRotation().RotateVector(FVector::ForwardVector).RotateAngleAxis(Angle, FVector::UpVector);
    FVector End = Start + Direction * 1000.0f;

    if (LineTraceSingle(Start, End, HitResult))
    {
        if (HitResult.Actor->ActorHasTag("Cover"))
        {
            // Found cover!
        }
    }
}
```

---

## Usage Examples

### Example 1: Low Health → Seek Cover

**Scenario:** Agent at 25% health, 2 enemies nearby

**MCTS Decision Process:**
1. Evaluates all 4 tactical actions + 4 legacy actions
2. `MoveToCoverAction`:
   - Urgency = 0.5 (health < 30%) + 0.2 (2 enemies) = 0.7
   - High priority
3. Selects `MoveToCoverAction`
4. Action calculates nearest cover position
5. Sets `Destination` in Blackboard
6. BT's MoveTo task executes pathfinding

**Expected Behavior:** Agent sprints to nearest cover

---

### Example 2: Healthy Agent → Attack Position

**Scenario:** Agent at 90% health, 1 enemy at 600 units

**MCTS Decision Process:**
1. Evaluates actions
2. `MoveToAttackPositionAction`:
   - Health > 80%, Stamina > 70%
   - Selects strategy: HighGround
3. Calculates elevated position around enemy
4. Sets `Destination`

**Expected Behavior:** Agent circles enemy to find high ground

---

### Example 3: Flanked → Reposition

**Scenario:** 2 enemies flanking at 90° and 270°, both within 400 units

**MCTS Decision Process:**
1. `TacticalRepositionAction`:
   - Detects flanking (2 enemies at wide angles)
   - Position quality = 0.2 (very poor)
   - Triggers reposition
2. Analyzes threat vectors from both enemies
3. Calculates safe direction (perpendicular to average threat)
4. Moves 600 units to safety

**Expected Behavior:** Agent repositions to avoid crossfire

---

## Testing and Debugging

### Enable Debug Logging

All movement actions log their decisions:

```cpp
UE_LOG(LogTemp, Display, TEXT("MoveToAllyAction: Moving to ally at distance %.2f"), Distance);
UE_LOG(LogTemp, Display, TEXT("MoveToAttackPositionAction: Strategy=%d"), Strategy);
```

**View Logs:**
- In Editor: Window → Output Log
- Filter by: `LogTemp`

### Visual Debugging

Add debug visualization in action implementations:

```cpp
#include "DrawDebugHelpers.h"

// In ExecuteAction()
DrawDebugSphere(World, Destination, 50.0f, 12, FColor::Green, false, 5.0f);
DrawDebugLine(World, MyPosition, Destination, FColor::Yellow, false, 5.0f);
```

### MCTS Visualization

Track which actions MCTS selects:

```cpp
// In UMoveToState::UpdateState()
UAction* SelectedAction = MCTS->GetBestAction();
UE_LOG(LogTemp, Warning, TEXT("MCTS Selected: %s"), *SelectedAction->GetName());
```

---

## Performance Considerations

### MCTS Complexity

**Current Setup:**
- 8 possible actions (4 tactical + 4 legacy)
- MCTS explores tree of depth 10
- Runs every frame in `UpdateState()`

**Optimization Options:**

1. **Reduce Action Count:**
   - Remove legacy directional actions (saves 50% search space)
   - Only include relevant actions based on context

2. **Throttle MCTS:**
   ```cpp
   // Run MCTS every N frames instead of every frame
   if (FrameCount % 5 == 0)
   {
       MCTS->RunMCTS(PossibleActions, StateMachine);
   }
   ```

3. **Async MCTS:**
   ```cpp
   // Run MCTS on background thread
   AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [this]() {
       MCTS->RunMCTS(PossibleActions, StateMachine);
   });
   ```

### Observation Updates

**Current:** 71 features updated every frame

**Optimization:**
- Update raycasts every 5 frames (they're expensive)
- Update enemy observations every 3 frames (perception system)
- Update cover only when moving

---

## Common Issues and Solutions

### Issue 1: "No valid ally destination found"

**Cause:** No allies tagged properly

**Solution:**
```cpp
// In ally character Blueprint
Tags: ["Ally"]
// OR in C++
AllyCharacter->Tags.Add(FName("Ally"));
```

---

### Issue 2: "No cover available"

**Cause:** Observation not updated with cover data

**Solution:**
```cpp
// In BT Service or Character Blueprint
bool bHasCover = PerformCoverCheck();
StateMachine->UpdateCoverObservation(bHasCover, Distance, Direction);
```

---

### Issue 3: Agent doesn't move

**Cause:** Behavior Tree not executing MoveTo task

**Solution:**
- Check Blackboard has "Destination" key (Vector)
- Check BT decorator enables MoveTo subtree when Strategy == "MoveTo"
- Verify StateMachine->SetDestination() is being called

---

### Issue 4: MCTS always selects same action

**Cause:** Reward function not distinguishing actions

**Solution:**
- Check observation data is being updated correctly
- Verify actions are calculating different destinations
- Increase MCTS exploration parameter

---

## Next Steps

### Recommended Enhancements

1. **Implement Behavior Tree Tasks:**
   - `BTTask_ExecuteMCTSDecision`
   - `BTTask_UpdateObservation`
   - `BTTask_CheckArrival`

2. **Create BT Services:**
   - `BTService_UpdateThreatAssessment`
   - `BTService_SyncObservationToBlackboard`

3. **Add EQS Queries:**
   - `EQS_FindCover` - Find optimal cover positions
   - `EQS_FindFlankingPosition` - Find flanking spots
   - `EQS_FindHighGround` - Find elevated positions

4. **Improve Cover System:**
   - Tag cover objects in level
   - Implement cover quality scoring
   - Dynamic cover destruction

5. **Multi-Agent Coordination:**
   - Team formations
   - Coordinated flanking
   - Shared threat assessment

---

## References

- **Main Documentation:** `CLAUDE.md`
- **Flee Implementation:** `FLEE_IMPLEMENTATION_GUIDE.md`
- **Observation System:** `Public/Core/ObservationElement.h`
- **MCTS Algorithm:** `Private/AI/MCTS.cpp`
- **Unreal Behavior Trees:** [UE5 Docs - Behavior Trees](https://docs.unrealengine.com/5.6/en-US/behavior-tree-in-unreal-engine/)
- **Unreal EQS:** [UE5 Docs - Environment Query System](https://docs.unrealengine.com/5.6/en-US/environment-query-system-in-unreal-engine/)

---

## Summary

The new tactical movement system transforms SBDAPM from simple directional movement to realistic 3D game AI behavior. The system:

✅ **Uses 71-feature observations** for informed decision-making
✅ **Provides 4 tactical actions** (Ally, Attack, Cover, Reposition)
✅ **Integrates with MCTS** for strategic planning
✅ **Works with Behavior Trees** for tactical execution
✅ **Adapts dynamically** to changing combat situations

The agent can now:
- Move to allies for support
- Find optimal attack positions (flanking, high ground)
- Seek cover when under fire
- Reposition when flanked or in poor positions

This creates emergent, intelligent movement behavior that resembles human player tactics in 3D combat games.

---

**Last Updated:** 2025-10-27
**Author:** Claude Code Assistant
**Version:** 2.0
