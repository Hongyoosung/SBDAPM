# Combat System Refactoring - Progress Tracker

**Status:** Sprint 1 Complete - Ready for Sprint 2
**Last Updated:** 2025-11-25

---

## Sprint 1: ✅ COMPLETED - Objective System Foundation

### Implemented Files

**Objective Base System:**
- ✅ `Public/Team/Objective.h` - Base objective class with lifecycle methods
- ✅ `Private/Team/Objective.cpp` - Base implementation

**Concrete Objectives:**
- ✅ `Public/Team/Objectives/EliminateObjective.h/cpp` - Kill enemy target
- ✅ `Public/Team/Objectives/CaptureObjective.h/cpp` - Capture zone/flag
- ✅ `Public/Team/Objectives/DefendObjective.h/cpp` - Defend zone/flag
- ✅ `Public/Team/Objectives/SupportAllyObjective.h/cpp` - Support ally teammate

**Objective Manager:**
- ✅ `Public/Team/ObjectiveManager.h` - Manages objective lifecycle and agent assignments
- ✅ `Private/Team/ObjectiveManager.cpp` - Full implementation with tick updates

**Integration:**
- ✅ `TeamLeaderComponent.h` - Added ObjectiveManager component and methods
- ✅ `TeamLeaderComponent.cpp` - Integrated objective queries and assignment

### Validation Needed
- [ ] Compile project
- [ ] Test objective creation (Blueprint or C++)
- [ ] Test agent assignment
- [ ] Test objective completion detection
- [ ] Test objective tick updates

---

## Sprint 2: 🔄 NEXT - Atomic Action Space (Weeks 1-2)

### Goal
Replace 16 discrete tactical actions with 8-dimensional continuous atomic action space

### Files to Create

**1. FTacticalAction Struct**
- **File:** `RL/RLTypes.h` (modify existing)
- **Content:**
```cpp
USTRUCT(BlueprintType)
struct FTacticalAction
{
    GENERATED_BODY()

    // Movement (continuous, 2D)
    UPROPERTY(BlueprintReadWrite)
    FVector2D MoveDirection = FVector2D::ZeroVector;  // [-1,1] x [-1,1]

    UPROPERTY(BlueprintReadWrite)
    float MoveSpeed = 1.0f;  // [0,1]

    // Aiming (continuous, 2D)
    UPROPERTY(BlueprintReadWrite)
    FVector2D LookDirection = FVector2D::ZeroVector;  // [-1,1] x [-1,1]

    // Discrete actions
    UPROPERTY(BlueprintReadWrite)
    bool bFire = false;

    UPROPERTY(BlueprintReadWrite)
    bool bCrouch = false;

    UPROPERTY(BlueprintReadWrite)
    bool bUseAbility = false;

    UPROPERTY(BlueprintReadWrite)
    int32 AbilityID = 0;
};

// Total: 8 dimensions
```

**2. Action Space Mask (Spatial Awareness)**
- **File:** `RL/RLTypes.h` (modify existing)
- **Content:**
```cpp
USTRUCT(BlueprintType)
struct FActionSpaceMask
{
    GENERATED_BODY()

    // Movement constraints
    UPROPERTY(BlueprintReadWrite)
    bool bLockMovementX = false;

    UPROPERTY(BlueprintReadWrite)
    bool bLockMovementY = false;

    UPROPERTY(BlueprintReadWrite)
    float MaxSpeed = 1.0f;

    // Aiming constraints
    UPROPERTY(BlueprintReadWrite)
    float MinYaw = -180.0f;

    UPROPERTY(BlueprintReadWrite)
    float MaxYaw = 180.0f;

    UPROPERTY(BlueprintReadWrite)
    float MinPitch = -90.0f;

    UPROPERTY(BlueprintReadWrite)
    float MaxPitch = 90.0f;

    // Action availability
    UPROPERTY(BlueprintReadWrite)
    bool bCanSprint = true;

    UPROPERTY(BlueprintReadWrite)
    bool bForceCrouch = false;

    UPROPERTY(BlueprintReadWrite)
    bool bSafetyLock = false;
};
```

**3. Modify RLPolicyNetwork**
- **Files:** `RL/RLPolicyNetwork.h/cpp`
- **Changes:**
  - Replace `ETacticalAction GetAction()` with `FTacticalAction GetAction()`
  - Add `GetAction(Observation, Objective, Mask)` overload
  - Update network output layer for 8D continuous actions

**4. Python Training Script**
- **File:** `Scripts/train_tactical_policy_v3.py` (new)
- **Changes:**
  - PPO with mixed continuous-discrete action space
  - Input: 71 features + 7 objective embedding + mask
  - Output: 8-dimensional action

### Tasks
- [ ] Add FTacticalAction struct to RLTypes.h
- [ ] Add FActionSpaceMask struct to RLTypes.h
- [ ] Modify RLPolicyNetwork.h to output FTacticalAction
- [ ] Implement RLPolicyNetwork.cpp atomic action logic
- [ ] Create train_tactical_policy_v3.py (PyTorch PPO)
- [ ] Test with random policy (no trained model)
- [ ] Compile and validate

---

## Sprint 3: StateTree Simplification (Weeks 3-4)

### Goal
Replace 5 complex execution tasks with single STTask_ExecuteObjective

### Files to Create

**1. STTask_ExecuteObjective**
- **Files:** `StateTree/Tasks/STTask_ExecuteObjective.h/cpp`
- **Replaces:** ExecuteAssault, ExecuteDefend, ExecuteSupport, ExecuteMove, ExecuteRetreat
- **Methods:**
  - `Tick()` - Main execution loop
  - `ExecuteMovement()` - Apply movement from atomic action
  - `ExecuteAiming()` - Apply aiming from atomic action
  - `ExecuteFire()` - Fire weapon
  - `ExecuteCrouch()` - Toggle crouch
  - `ApplyMask()` - Apply spatial constraints

**2. STEvaluator_SpatialContext**
- **Files:** `StateTree/Evaluators/STEvaluator_SpatialContext.h/cpp`
- **Purpose:** Compute action space mask based on environment
- **Updates:** Every frame, writes FActionSpaceMask to context
- **Methods:**
  - `DetectIndoor()` - Check if in indoor space
  - `MeasureLateralClearance()` - Check corridor width
  - `MeasureNavMeshEdgeDistance()` - Check cliff proximity
  - `ApplyCoverAimingRestrictions()` - Limit aim angles at cover

**3. Simplified StateTree Structure**
- **Change:** 6 states → 3 states
  - Idle (wait for objective)
  - Active (execute objective)
  - Dead (terminal)

### Files to Delete
- [ ] `StateTree/Tasks/STTask_ExecuteAssault.h/cpp` (500+ lines)
- [ ] `StateTree/Tasks/STTask_ExecuteDefend.h/cpp`
- [ ] `StateTree/Tasks/STTask_ExecuteSupport.h/cpp`
- [ ] `StateTree/Tasks/STTask_ExecuteMove.h/cpp`
- [ ] `StateTree/Tasks/STTask_ExecuteRetreat.h/cpp`

### Tasks
- [ ] Implement STTask_ExecuteObjective
- [ ] Implement STEvaluator_SpatialContext
- [ ] Update FollowerStateTreeSchema for new context
- [ ] Create Blueprint StateTree asset with 3 states
- [ ] Test objective execution
- [ ] Delete old tasks after validation

---

## Sprint 4: Hierarchical Rewards (Weeks 5-6)

### Goal
Unify individual, coordination, and strategic rewards

### Files to Create

**RewardCalculator Component**
- **Files:** `RL/RewardCalculator.h/cpp`
- **Methods:**
  - `CalculateTotalReward()` - Sum all reward components
  - `CalculateIndividualReward()` - Base combat rewards
  - `CalculateCoordinationReward()` - Teamwork bonuses
  - `CalculateObjectiveReward()` - Strategic progress
  - `CalculateEfficiencyPenalty()` - Wasted actions

**Reward Structure:**
```cpp
// Individual (existing)
+10  Kill
+5   Damage
-5   Take Damage
-10  Death

// Coordination (NEW)
+15  Kill while on objective
+10  Combined fire (2+ agents hit same enemy)
+5   Formation maintenance
-15  Disobey objective

// Strategic (NEW, for MCTS)
+50  Objective complete
+30  Enemy squad eliminated
-30  Own squad eliminated
```

### Tasks
- [ ] Implement RewardCalculator component
- [ ] Add coordination detection (combined fire tracking)
- [ ] Integrate with FollowerAgentComponent
- [ ] Update MCTS backpropagation for objective rewards
- [ ] Test reward alignment

---

## Sprint 5: MCTS Objective Selection (Weeks 7-8)

### Goal
MCTS selects objectives instead of command combinations (14,641 → ~50 actions)

### Files to Modify

**MCTS.cpp Changes:**
- Replace `GenerateCommandCombinations()` with `GenerateObjectiveAssignments()`
- Action space: 7 objective types × N agents = ~50 combinations (vs 14,641)
- Use UCB for objective selection

**TeamLeaderComponent Changes:**
- MCTS outputs: `TMap<AActor*, UObjective*>` (objective assignments)
- Agents query objective from ObjectiveManager
- Remove old FStrategicCommand system (or keep for compatibility)

### Tasks
- [ ] Modify MCTS action generation
- [ ] Update MCTS simulation to use objectives
- [ ] Integrate with ObjectiveManager
- [ ] Benchmark MCTS performance (target: <15ms)
- [ ] Test objective-driven decision making

---

## Sprint 6: End-to-End Validation (Weeks 9-10)

### Goal
Complete integration and performance validation

### Validation Scenarios
1. **1v1 Elimination:** Single agent eliminates enemy
2. **2v2 Capture:** Two agents capture zone
3. **4v4 Defense:** Team defends objective from enemy squad
4. **Mixed Objectives:** Multiple objectives active simultaneously

### Performance Targets
- MCTS: 15-20ms (down from 34ms)
- RL Inference: 1-3ms (unchanged)
- Spatial Evaluator: <0.5ms
- Total Frame Budget: <10ms

### Tasks
- [ ] Create test scenarios in Blueprint
- [ ] Run performance profiling (Unreal Insights)
- [ ] Validate objective completion detection
- [ ] Test reward calculation accuracy
- [ ] Document any issues or limitations

---

## File Change Summary

### New Files (19)
```
Team/Objective.h/cpp
Team/Objectives/EliminateObjective.h/cpp
Team/Objectives/CaptureObjective.h/cpp
Team/Objectives/DefendObjective.h/cpp
Team/Objectives/SupportAllyObjective.h/cpp
Team/ObjectiveManager.h/cpp
StateTree/Tasks/STTask_ExecuteObjective.h/cpp
StateTree/Evaluators/STEvaluator_SpatialContext.h/cpp
RL/RewardCalculator.h/cpp
Scripts/train_tactical_policy_v3.py
```

### Modified Files (8)
```
RL/RLTypes.h                    # FTacticalAction, FActionSpaceMask
RL/RLPolicyNetwork.h/cpp        # Atomic action output
Team/TeamLeaderComponent.h/cpp  # ObjectiveManager integration
Team/FollowerAgentComponent.h/cpp # RewardCalculator integration
AI/MCTS/MCTS.h/cpp              # Objective selection
StateTree/FollowerStateTreeSchema.h # Context update
```

### Deleted Files (10)
```
StateTree/Tasks/STTask_ExecuteAssault.h/cpp
StateTree/Tasks/STTask_ExecuteDefend.h/cpp
StateTree/Tasks/STTask_ExecuteSupport.h/cpp
StateTree/Tasks/STTask_ExecuteMove.h/cpp
StateTree/Tasks/STTask_ExecuteRetreat.h/cpp
```

---

## Current Blockers

**None** - Sprint 1 complete, ready for Sprint 2

---

## Next Immediate Steps

1. **Compile Sprint 1** - Validate objective system compiles
2. **Test Objective Creation** - Create simple test scenario
3. **Start Sprint 2** - Define FTacticalAction struct

---

## References

- **Main Plan:** `REFACTORING_COMBATSYSTEM_PLAN.md`
- **AI Integration:** `REFACTORING_AISYSTEM_PLAN.md`
- **Architecture:** `CLAUDE.md`
