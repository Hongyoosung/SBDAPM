# Combat System Refactoring Plan

## Current System Analysis

### Problems Identified

**1. Two-Layer Action Complexity**
```
Strategic Commands (15 types) → Follower States (6 types) → Tactical Actions (16 types) → Task Logic
```
- Strategic: Assault, Defend, Support, MoveTo, Retreat, Patrol, Advance, Regroup, TakeCover, etc.
- Tactical: AggressiveAssault, CautiousAdvance, FlankLeft, FlankRight, SeekCover, SuppressiveFire, etc.
- Each tactical action requires custom implementation in StateTree tasks
- Heavy switch statements in `STTask_ExecuteAssault.cpp`, `STTask_ExecuteDefend.cpp`

**2. Unclear Separation of Concerns**
- Strategic commands control WHAT to do (objective)
- Tactical actions control HOW to do it (execution details)
- But both layers make similar decisions (positioning, aggression level)
- Example: "Assault" (strategic) + "CautiousAdvance" (tactical) overlap conceptually

**3. Reward Misalignment**
```cpp
// Current rewards in RLTypes.h:220
KILL_ENEMY = 10.0f;         // Individual
REACH_COVER = 5.0f;         // Individual
FOLLOW_COMMAND = 2.0f;      // Weak coordination signal
```
- No rewards for objectives (capture base, protect ally)
- No team-level coordination bonuses
- Strategic layer (MCTS) and tactical layer (RL) optimize different things

**4. Difficult to Extend**
- Adding new objective type (e.g., "Rescue Ally") requires:
  - New strategic command enum
  - New tactical action enum
  - New StateTree task implementation
  - New reward logic scattered across files
- High development friction for gameplay features

**5. Incompatible with AI Refactoring (v3.0)**
- World model must predict 16 tactical actions × N followers = huge state space
- Value network struggles with ambiguous strategic commands
- MCTS explores 14,641 command combinations, many redundant

---

## Refactored Architecture

### Design Principles

1. **Strategic = Objectives**, Tactical = Low-Level Control
2. **Objectives are explicit** (not implicit in commands)
3. **Atomic actions** for world model prediction
4. **Hierarchical rewards** align both layers
5. **Extensible** for new objective types

---

### New Action Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGIC LAYER (Team Leader)                │
│                                                                 │
│  Objectives:                                                    │
│    • Eliminate(Target)         - Focus fire on enemy            │
│    • CaptureObjective(Zone)    - Secure objective area          │
│    • DefendObjective(Zone)     - Hold objective area            │
│    • SupportAlly(Ally)         - Assist specific teammate       │
│    • FormationMove(Location)   - Coordinated movement           │
│    • Retreat(Location)         - Fall back to position          │
│                                                                 │
│  Each objective includes:                                       │
│    • Target (Actor/Location)                                    │
│    • Priority (0-10)                                            │
│    • Success condition (delegate)                               │
│    • Time limit (optional)                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TACTICAL LAYER (Follower RL)                 │
│                                                                 │
│  Primitive Actions (Atomic, easily predictable):                │
│    • Move(Direction, Speed)       - Navigate                    │
│    • LookAt(Target)               - Aim                         │
│    • Fire(Continuous/Burst)       - Shoot weapon                │
│    • Crouch/Stand                 - Stance                      │
│    • UseAbility(ID)               - Special action              │
│    • Wait                         - Do nothing                  │
│                                                                 │
│  RL Policy outputs:                                              │
│    • Movement vector (2D)                                        │
│    • Look direction (2D)                                         │
│    • Discrete action (Fire/Crouch/Ability/Wait)                 │
│                                                                 │
│  Context-aware selection:                                        │
│    • Current objective from leader                               │
│    • Local observations (enemies, cover, allies)                 │
│    • Learned tactical behavior                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### Phase 1: Objective System (Strategic Layer)

#### 1.1 Objective Base Class
**New File:** `Team/Objective.h/cpp`

```cpp
UENUM(BlueprintType)
enum class EObjectiveType : uint8
{
    Eliminate,           // Kill specific enemy
    CaptureObjective,    // Capture zone/flag
    DefendObjective,     // Hold zone/flag
    SupportAlly,         // Provide covering fire
    FormationMove,       // Coordinated movement
    Retreat              // Fall back
};

UCLASS(Abstract, Blueprintable)
class UObjective : public UObject
{
    GENERATED_BODY()

public:
    // Core properties
    UPROPERTY(BlueprintReadWrite)
    EObjectiveType Type;

    UPROPERTY(BlueprintReadWrite)
    AActor* TargetActor = nullptr;

    UPROPERTY(BlueprintReadWrite)
    FVector TargetLocation = FVector::ZeroVector;

    UPROPERTY(BlueprintReadWrite)
    int32 Priority = 5;  // 0-10

    UPROPERTY(BlueprintReadWrite)
    float TimeLimit = 0.0f;  // 0 = no limit

    UPROPERTY(BlueprintReadWrite)
    TArray<AActor*> AssignedAgents;

    // State
    UPROPERTY(BlueprintReadOnly)
    bool bIsActive = false;

    UPROPERTY(BlueprintReadOnly)
    bool bIsCompleted = false;

    UPROPERTY(BlueprintReadOnly)
    float Progress = 0.0f;  // 0.0-1.0

    // Methods
    virtual void Activate();
    virtual void Deactivate();
    virtual bool IsCompleted() const;
    virtual float GetProgress() const;
    virtual float CalculateReward() const;  // Strategic reward for MCTS
};
```

#### 1.2 Concrete Objective Types
**New Files:**
- `Team/Objectives/EliminateObjective.h/cpp`
- `Team/Objectives/CaptureObjective.h/cpp`
- `Team/Objectives/DefendObjective.h/cpp`
- `Team/Objectives/SupportAllyObjective.h/cpp`
- `Team/Objectives/RescueAllyObjective.h/cpp`

Example:
```cpp
UCLASS()
class UEliminateObjective : public UObjective
{
    GENERATED_BODY()

public:
    UEliminateObjective() { Type = EObjectiveType::Eliminate; }

    virtual bool IsCompleted() const override
    {
        return TargetActor && !IsValid(TargetActor); // Target dead
    }

    virtual float CalculateReward() const override
    {
        if (IsCompleted()) return 50.0f;  // Strategic kill reward

        // Partial credit for damage dealt
        if (UHealthComponent* Health = TargetActor->FindComponentByClass<UHealthComponent>())
        {
            return (1.0f - Health->GetHealthPercent()) * 25.0f;
        }
        return 0.0f;
    }
};
```

#### 1.3 Objective Manager
**New File:** `Team/ObjectiveManager.h/cpp`

```cpp
UCLASS()
class UObjectiveManager : public UActorComponent
{
    GENERATED_BODY()

public:
    // Create objective
    UFUNCTION(BlueprintCallable)
    UObjective* CreateObjective(EObjectiveType Type, AActor* Target, int32 Priority);

    // Assign agents to objective
    UFUNCTION(BlueprintCallable)
    void AssignAgentsToObjective(UObjective* Objective, const TArray<AActor*>& Agents);

    // Query active objectives
    UFUNCTION(BlueprintPure)
    TArray<UObjective*> GetActiveObjectives() const;

    // Get agent's current objective
    UFUNCTION(BlueprintPure)
    UObjective* GetAgentObjective(AActor* Agent) const;

private:
    UPROPERTY()
    TArray<UObjective*> ActiveObjectives;

    UPROPERTY()
    TMap<AActor*, UObjective*> AgentObjectiveMap;
};
```

**Modified Files:**
- `Team/TeamLeaderComponent.h` - Add `UObjectiveManager* ObjectiveManager`
- `AI/MCTS/MCTS.cpp` - MCTS selects objectives instead of command combinations

---

### Phase 2: Simplified Tactical Actions

#### 2.1 Atomic Action Space
**Modified File:** `RL/RLTypes.h`

```cpp
// REPLACE old ETacticalAction (16 enums) with:
USTRUCT(BlueprintType)
struct FTacticalAction
{
    GENERATED_BODY()

    // Movement (continuous)
    UPROPERTY(BlueprintReadWrite)
    FVector2D MoveDirection = FVector2D::ZeroVector;  // [-1,1] x [-1,1]

    UPROPERTY(BlueprintReadWrite)
    float MoveSpeed = 1.0f;  // [0,1] - percentage of max speed

    // Aiming (continuous)
    UPROPERTY(BlueprintReadWrite)
    FVector2D LookDirection = FVector2D::ZeroVector;  // [-1,1] x [-1,1]

    // Discrete actions (one-hot)
    UPROPERTY(BlueprintReadWrite)
    bool bFire = false;

    UPROPERTY(BlueprintReadWrite)
    bool bCrouch = false;

    UPROPERTY(BlueprintReadWrite)
    bool bUseAbility = false;

    UPROPERTY(BlueprintReadWrite)
    int32 AbilityID = 0;
};

// Action space size: 2 (move) + 1 (speed) + 2 (look) + 3 (discrete) = 8 dimensions
// Much simpler for world model prediction!
```

#### 2.2 RL Policy Output
**Modified File:** `RL/RLPolicyNetwork.h/cpp`

```cpp
class URLPolicyNetwork : public UObject
{
    GENERATED_BODY()

public:
    // OLD: TArray<float> GetActionProbabilities(const FObservationElement& Obs)
    // Returns [16] action probabilities

    // NEW: Get atomic action
    UFUNCTION(BlueprintCallable)
    FTacticalAction GetAction(const FObservationElement& Obs, UObjective* CurrentObjective);

    // For MCTS priors (Phase 3-4 of AI refactoring)
    UFUNCTION(BlueprintCallable)
    TArray<float> GetObjectivePriors(const FTeamObservation& TeamObs);
    // Returns [7] prior probabilities (one per objective type)
};
```

**Python Training Changes:**
- Input: 71 features (observation) + 7 features (current objective embedding)
- Output: 8-dimensional action (move_x, move_y, speed, look_x, look_y, fire, crouch, ability)
- Loss: PPO with mixed continuous-discrete action space

---

### Phase 3: StateTree Simplification

#### 3.1 Remove Redundant Tasks
**Delete Files:**
- `STTask_ExecuteAssault.h/cpp` (500+ lines of switch logic)
- `STTask_ExecuteDefend.h/cpp`
- `STTask_ExecuteSupport.h/cpp`
- `STTask_ExecuteMove.h/cpp`
- `STTask_ExecuteRetreat.h/cpp`

**Replace With Single Task:** `STTask_ExecuteObjective.h/cpp`

```cpp
USTRUCT(meta = (DisplayName = "Execute Objective"))
struct FSTTask_ExecuteObjective : public FStateTreeTaskBase
{
    GENERATED_BODY()

    virtual EStateTreeRunStatus Tick(FStateTreeExecutionContext& Context, float DeltaTime) const override
    {
        FFollowerStateTreeContext& SharedContext = GetSharedContext(Context);

        // 1. Get current objective from leader
        UObjective* CurrentObjective = SharedContext.TeamLeader->GetObjectiveForFollower(SharedContext.FollowerComponent);

        // 2. Query RL policy (includes objective context)
        FTacticalAction Action = SharedContext.TacticalPolicy->GetAction(
            SharedContext.CurrentObservation,
            CurrentObjective
        );

        // 3. Execute atomic actions (simple, no switch logic!)
        ExecuteMovement(Context, Action.MoveDirection, Action.MoveSpeed);
        ExecuteAiming(Context, Action.LookDirection);

        if (Action.bFire) ExecuteFire(Context);
        if (Action.bCrouch) ExecuteCrouch(Context);
        if (Action.bUseAbility) ExecuteAbility(Context, Action.AbilityID);

        // 4. Calculate hierarchical reward
        float Reward = CalculateReward(Context, CurrentObjective, Action, DeltaTime);
        SharedContext.FollowerComponent->ProvideReward(Reward);

        return EStateTreeRunStatus::Running;
    }

private:
    void ExecuteMovement(FStateTreeExecutionContext& Context, FVector2D Direction, float Speed) const;
    void ExecuteAiming(FStateTreeExecutionContext& Context, FVector2D Direction) const;
    void ExecuteFire(FStateTreeExecutionContext& Context) const;
    void ExecuteCrouch(FStateTreeExecutionContext& Context) const;
    void ExecuteAbility(FStateTreeExecutionContext& Context, int32 AbilityID) const;
    float CalculateReward(FStateTreeExecutionContext& Context, UObjective* Objective, const FTacticalAction& Action, float DeltaTime) const;
};
```

#### 3.2 Simplified StateTree Structure
**Before (v2.0):**
```
States: Idle → Assault → Defend → Support → Move → Retreat → Dead
Each state has different task logic
```

**After (v3.0):**
```
States: Idle → Active → Dead
- Idle: Wait for objective assignment
- Active: Execute any objective (single task handles all types)
- Dead: Terminal state
```

---

### Phase 4: Hierarchical Reward System

#### 4.1 New Reward Calculator
**New File:** `RL/RewardCalculator.h/cpp`

```cpp
UCLASS()
class URewardCalculator : public UObject
{
    GENERATED_BODY()

public:
    // Calculate total reward (individual + coordination + strategic)
    UFUNCTION(BlueprintCallable)
    float CalculateTotalReward(
        AActor* Agent,
        UObjective* CurrentObjective,
        const FTacticalAction& Action,
        float DeltaTime
    );

private:
    // Individual rewards (base RL)
    float CalculateIndividualReward(AActor* Agent, const FTacticalAction& Action, float DeltaTime);
    // +10 kill, +5 damage, -5 take damage, -10 death

    // Coordination rewards (NEW)
    float CalculateCoordinationReward(AActor* Agent, UObjective* Objective, const FTacticalAction& Action);
    // +15 kill while on objective
    // +10 combined fire (2+ agents hit same enemy within 1s)
    // +5 formation maintenance
    // -15 disobey objective (wrong positioning)

    // Strategic rewards (objective progress, for MCTS backprop)
    float CalculateObjectiveReward(UObjective* Objective);
    // +50 objective complete
    // +30 enemy squad eliminated
    // -30 own squad eliminated

    // Efficiency penalties
    float CalculateEfficiencyPenalty(AActor* Agent, const FTacticalAction& Action);
    // -1 wasted ammo (miss shots)
    // -2 out of position
    // -1 idle too long
};
```

#### 4.2 Reward Propagation
```
Individual Reward → RL Policy (PPO training)
    ↓
Coordination Reward → RL Policy (align with objectives)
    ↓
Strategic Reward → MCTS Backpropagation → Value Network training
```

**Modified Files:**
- `Team/FollowerAgentComponent.cpp:OnDamageTakenEvent()` - Use RewardCalculator
- `Team/TeamLeaderComponent.cpp:BackpropagateReward()` - Objective rewards
- `AI/MCTS/MCTS.cpp:BackpropagateNode()` - Team-level rewards

---

## Benefits of Refactored System

### 1. Clearer Separation of Concerns
| Layer | Responsibility | Examples |
|-------|----------------|----------|
| **Strategic (MCTS)** | WHAT to do (objectives) | "Eliminate Enemy A", "Capture Zone B" |
| **Tactical (RL)** | HOW to do it (low-level control) | Move, Aim, Fire, Crouch |

### 2. Simplified Action Space
| Before | After |
|--------|-------|
| 15 strategic commands × 16 tactical actions = 240 combinations | 7 objectives × 8-dim continuous actions |
| Switch statements with 500+ lines per task | Single task with atomic action executors |
| Hard to add new behaviors | Add new objective class, done |

### 3. World Model Compatibility
**Before:** Predict 16 discrete tactical actions per follower
```python
WorldModel.predict(state, [Assault+FlankLeft, Defend+SeekCover, Support+CoveringFire, ...])
# 16^N state space explosion
```

**After:** Predict 8-dimensional continuous action
```python
WorldModel.predict(state, [(move_x, move_y, speed, look_x, look_y, fire, crouch, ability), ...])
# Smooth continuous space, easier regression
```

### 4. Reward Alignment
**Before:** Individual rewards ≠ Strategic rewards
- RL optimizes for kills
- MCTS optimizes for objectives
- Conflicting objectives

**After:** Hierarchical rewards aligned
- Base: Individual performance
- Bonus: Coordination with objective
- Team: Objective completion
- All layers optimize same goal

### 5. Extensibility
**Adding new objective (e.g., "Hack Terminal"):**

Before:
1. Add `EStrategicCommandType::HackTerminal`
2. Add tactical actions for hacking
3. Implement `STTask_ExecuteHack.cpp` (200+ lines)
4. Update reward logic in multiple files
5. Retrain RL policy with new action space

After:
1. Create `UHackTerminalObjective.cpp` (50 lines)
2. Register with `ObjectiveManager`
3. Done! RL already handles it (atomic actions unchanged)

---

## Migration Path

### Sprint 1 (Week 1): Objective System Foundation
- [ ] Implement `Objective.h/cpp` base class
- [ ] Implement concrete objective types (Eliminate, Capture, Defend)
- [ ] Implement `ObjectiveManager.h/cpp`
- [ ] Add to `TeamLeaderComponent`
- [ ] Unit tests for objective creation/completion

**Validation:** Can create objectives, assign agents, query progress

### Sprint 2 (Week 2): Atomic Action Space
- [ ] Define `FTacticalAction` struct
- [ ] Modify `RLPolicyNetwork` to output atomic actions
- [ ] Update Python training script (continuous-discrete action space)
- [ ] Test with dummy policy (random actions)

**Validation:** Agents can move/aim/fire with new action format

### Sprint 3 (Week 3): StateTree Simplification
- [ ] Implement `STTask_ExecuteObjective.cpp`
- [ ] Implement atomic action executors (ExecuteMovement, ExecuteAiming, etc.)
- [ ] Simplify StateTree to 3 states (Idle, Active, Dead)
- [ ] Remove old Execute tasks (Assault, Defend, Support, Move, Retreat)

**Validation:** Agents execute objectives with new task

### Sprint 4 (Week 4): Hierarchical Rewards
- [ ] Implement `RewardCalculator.h/cpp`
- [ ] Add coordination detection (combined fire, formation)
- [ ] Integrate with `FollowerAgentComponent`
- [ ] Update MCTS backpropagation for objective rewards

**Validation:** Reward breakdown shows individual + coordination + strategic

### Sprint 5 (Week 5): MCTS Integration
- [ ] Modify `MCTS.cpp` to select objectives instead of command combinations
- [ ] Reduce action space from 14,641 → ~50 (7 objectives × ~7 combinations)
- [ ] Benchmark MCTS performance (should be faster)

**Validation:** MCTS selects objectives, agents execute them

### Sprint 6 (Week 6): Training & Testing
- [ ] Retrain RL policy with objective context
- [ ] Test all objective types (Eliminate, Capture, Defend, Support, Rescue)
- [ ] Performance profiling (ensure <10ms frame budget)
- [ ] Validation scenarios (1v1, 2v2, 4v4)

**Validation:** Trained agents complete objectives successfully

---

## Compatibility with AI System Refactoring

### Phase 1-2 (Value Network + World Model)
✅ **Compatible** - Atomic actions simplify state prediction
- World model predicts 8-dim action space (easy regression)
- Value network evaluates objective progress (clear signal)

### Phase 3-4 (Coupled Training)
✅ **Compatible** - Objectives provide natural MCTS priors
- RL policy outputs objective priors: `P(Eliminate) = 0.6, P(Defend) = 0.3, ...`
- MCTS initializes nodes with these priors
- CurriculumManager prioritizes high-uncertainty objectives

### Phase 5 (Reward Alignment)
✅ **Already Implemented** - Hierarchical rewards built-in
- Individual + Coordination + Strategic rewards
- All layers optimize aligned objectives

### Phase 6 (Continuous Planning)
✅ **Compatible** - Objectives persist across planning cycles
- MCTS can replan without invalidating current objective
- Smooth transitions between objectives

---

## Risk Mitigation

**Risk 1: Continuous action space harder to learn**
- **Mitigation:** Hybrid discrete-continuous PPO (proven in MuJoCo, OpenAI Gym)
- **Fallback:** Discretize continuous actions (8 directions, 3 speeds)

**Risk 2: Objective system adds overhead**
- **Mitigation:** Lightweight UObject, minimal per-frame cost
- **Profiling:** Budget 0.5ms for objective queries

**Risk 3: Breaking existing content**
- **Mitigation:** Phased migration (keep old system running in parallel)
- **Testing:** Regression tests for all scenarios

**Risk 4: RL policy needs retraining from scratch**
- **Mitigation:** Transfer learning (reuse low-level skills)
- **Expected:** 200-300 training episodes to converge (vs 500+ from scratch)

---

## Performance Comparison (Estimated)

| Metric | Before (v2.0) | After (v3.0) |
|--------|---------------|--------------|
| **MCTS Action Space** | 14,641 combinations | ~50 objective assignments |
| **MCTS Simulation Time** | 34ms (1000 sims) | **~15ms** (same quality, fewer nodes) |
| **RL Action Complexity** | 16 discrete → switch logic | 8-dim continuous → direct execution |
| **StateTree Task Count** | 7 tasks (500+ lines each) | **1 task** (200 lines total) |
| **World Model Prediction** | 16^N discrete space | **8-dim regression** |
| **Lines of Code** | ~3,500 (task logic) | **~1,200** (65% reduction) |
| **Time to Add Feature** | 4-6 hours (5 files) | **30 min** (1 file) |

---

## Success Metrics

**Quantitative:**
1. **MCTS Speed:** 2x faster simulation (15ms vs 34ms)
2. **Code Reduction:** 65% fewer lines in combat logic
3. **Training Speed:** Converge in ≤300 episodes (vs 500+ baseline)
4. **Objective Completion:** ≥80% success rate in test scenarios

**Qualitative:**
1. **Clear Intent:** Objectives explicit in logs/debug UI
2. **Smooth Execution:** No jerky transitions between actions
3. **Emergent Tactics:** Combined fire, cover usage, rescue behavior
4. **Developer Velocity:** New objectives in <1 hour

---

## Files Changed Summary

### New Files (14)
```
Team/Objective.h/cpp                        # Base objective class
Team/Objectives/EliminateObjective.h/cpp    # Kill target
Team/Objectives/CaptureObjective.h/cpp      # Capture zone
Team/Objectives/DefendObjective.h/cpp       # Defend zone
Team/Objectives/SupportAllyObjective.h/cpp  # Support teammate
Team/Objectives/RescueAllyObjective.h/cpp   # Rescue wounded
Team/ObjectiveManager.h/cpp                 # Objective lifecycle
StateTree/Tasks/STTask_ExecuteObjective.h/cpp  # Unified execution task
RL/RewardCalculator.h/cpp                   # Hierarchical rewards
```

### Modified Files (8)
```
RL/RLTypes.h                                # FTacticalAction struct
RL/RLPolicyNetwork.h/cpp                    # Atomic action output
Team/TeamLeaderComponent.h/cpp              # Objective integration
Team/FollowerAgentComponent.h/cpp           # Reward calculator
AI/MCTS/MCTS.h/cpp                          # Objective selection
StateTree/FollowerStateTreeSchema.h         # Context update
```

### Deleted Files (10)
```
StateTree/Tasks/STTask_ExecuteAssault.h/cpp     # Replaced by ExecuteObjective
StateTree/Tasks/STTask_ExecuteDefend.h/cpp
StateTree/Tasks/STTask_ExecuteSupport.h/cpp
StateTree/Tasks/STTask_ExecuteMove.h/cpp
StateTree/Tasks/STTask_ExecuteRetreat.h/cpp
```

**Net Change:** +14 new, +8 modified, -10 deleted = **+12 files total**

---


#### 6. Alignment with Industry Practice
**AlphaZero / MuZero Development:**
1. Define game rules first (equivalent to our combat system)
2. Build AI on top (MCTS + neural networks)
3. Never change game rules mid-training

**OpenAI Five / Dota 2:**
1. Dota 2 game mechanics stable
2. Train RL agents on stable mechanics
3. When Dota patches change mechanics → Retrain agents

**Lesson:** Get environment/actions right first, then optimize AI.

---

### Recommended Implementation Order

#### **Phase 1: Combat System (Weeks 1-6)** ← START HERE
Follow the 6 sprints in Combat Refactoring Plan:
1. Objective system foundation
2. Atomic action space
3. StateTree simplification
4. Hierarchical rewards
5. MCTS integration
6. Training & testing

**Deliverable:** Stable combat system with objectives, atomic actions, hierarchical rewards

**Validation:** Agents execute objectives successfully with rule-based MCTS (v2.0 baseline)

---

#### **Phase 2: AI System (Weeks 7-20)** ← AFTER COMBAT STABLE
Follow the 7 sprints in AI Refactoring Plan:
1. Value Network (guides MCTS with objective values)
2. World Model (predicts atomic action outcomes)
3. MCTS → RL curriculum (prioritize hard objectives)
4. RL → MCTS priors (objective probabilities)
5. Reward alignment (already done in Phase 1!)
6. Continuous planning
7. Self-play pipeline

**Deliverable:** AlphaZero-inspired AI with learned value/world models, coupled MCTS+RL

**Validation:** v3.0 agents beat v2.0 baseline ≥70% in 4v4 scenarios

---

### Migration Strategy (Minimal Disruption)

**Step 1: Create parallel combat system (Week 1)**
- New objective system coexists with old command system
- Feature flag: `bUseObjectiveSystem = false` (default off)
- Both systems functional, can A/B test

**Step 2: Validate new system (Weeks 2-5)**
- Test new objective system thoroughly
- Fix bugs, performance issues
- Keep old system as fallback

**Step 3: Switch default (Week 6)**
- `bUseObjectiveSystem = true` (default on)
- Monitor for issues
- Can toggle back if needed

**Step 4: Remove old system (Week 7)**
- Once confident, delete old command/tactical code
- Clean up codebase
- Start AI refactoring on stable foundation

---


## Conclusion

**Recommendation: Refactor Combat System First**

**Timeline:**
- **Weeks 1-6:** Combat System Refactoring (this plan)
- **Weeks 7-20:** AI System Refactoring (REFACTORING_AISYSTEM_PLAN.md)
- **Total:** 20 weeks (vs 24 weeks if AI first)

**Next Steps:**
1. Review and approve this combat refactoring plan
2. Assign team (2-3 engineers)
3. Start Sprint 1 (Objective System Foundation)
4. Complete all 6 sprints
5. Validate stable combat system
6. Begin AI refactoring on stable foundation

**Success Criteria Before Moving to AI Refactoring:**
- [ ] All objective types implemented (Eliminate, Capture, Defend, Support, Rescue)
- [ ] Atomic actions execute smoothly (move, aim, fire, crouch)
- [ ] StateTree simplified to 3 states
- [ ] Hierarchical rewards working (individual + coordination + strategic)
- [ ] MCTS selects objectives efficiently (<15ms)
- [ ] Agents complete objectives ≥80% success rate in test scenarios
- [ ] Performance budget met (<10ms per frame)
- [ ] Code reduction achieved (65% fewer lines)

Once these criteria are met, proceed to AI refactoring with confidence.
