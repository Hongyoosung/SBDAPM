# EQS Cover System - Integration Guide

## Architecture Overview

**Hierarchical AI System:**
```
Team Leader (MCTS) → Strategic Command (Assault, Defend, Support, Move)
    ↓
Follower (RL Policy) → Tactical Action (SeekCover, DefensiveHold, etc.)
    ↓
Behavior Tree → Execute Action (tag-based or EQS-based cover finding)
```

**Cover Finding Flow:**
1. Team Leader issues "Defend" command
2. Follower RL policy selects "SeekCover" tactical action
3. BT executes cover finding (tag-based or EQS)

---

## 1. EQS Components (Already Implemented)

### C++ Classes
- `UEnvQueryGenerator_CoverPoints` - Generates cover candidates
- `UEnvQueryTest_CoverQuality` - Evaluates cover quality
- `UEnvQueryContext_CoverEnemies` - Provides enemy positions from Team Leader
- `UBTTask_FindCoverLocation` - BT task for EQS-based cover finding

### Current Cover Finding Methods

**Method A: Tag-Based (Default)**
- Used by: `BTTask_ExecuteDefend::FindNearestCover()`
- Searches for actors with "Cover" tag
- Simple distance-based selection

**Method B: EQS-Based (Advanced)**
- Used by: `BTTask_FindCoverLocation`
- Multi-factor evaluation (enemy distance, LOS, navigability)
- Integrates with Team Leader enemy tracking

---

## 2. Editor Setup

### Step 1: Create EQS Asset
1. **Content Browser** → Right-click → **Artificial Intelligence** → **Environment Query**
2. Name: `EQS_FindCover`

### Step 2: Configure Generator
**Add Generator:** `EnvQueryGenerator_CoverPoints`
```
SearchRadius: 1500.0         // 15m search radius
GridSpacing: 200.0           // 2m between samples
CoverTag: "Cover"            // Tag for pre-placed cover
bIncludeTaggedCover: true    // Use tagged actors
bGenerateGridPoints: true    // Grid sampling
```

### Step 3: Configure Test
**Add Test:** `EnvQueryTest_CoverQuality`
```
EnemyContext: EnvQueryContext_CoverEnemies  // Links to Team Leader
EnemyDistanceWeight: 0.5     // 50% - prefer farther from enemies
LineOfSightWeight: 0.3       // 30% - prefer LOS blocking
QuerierDistanceWeight: 0.2   // 20% - prefer closer to agent
MinSafeDistance: 500.0       // 5m minimum
MaxSafeDistance: 2000.0      // 20m maximum
bCheckNavigability: true
bCheckLineOfSight: true
```

### Step 4: Tag Cover Objects
1. Select static meshes (walls, crates, etc.)
2. **Details Panel** → **Tags** → Add: `Cover`

---

## 3. Behavior Tree Integration

### Current BT Structure
```
Root
├─ [CheckCommandType: Defend] Defend Subtree
│  ├─ Query RL Policy (periodic)
│  └─ Execute Defend Task
│     ├─ [TacticalAction: SeekCover] → FindNearestCover() [tag-based]
│     ├─ [TacticalAction: DefensiveHold] → Hold position
│     └─ [TacticalAction: SuppressiveFire] → Fire from position
└─ [CheckCommandType: Assault] Assault Subtree
   └─ ...
```

### Integration Options

**Option A: Replace Tag-Based with EQS (Recommended)**
Modify `BTTask_ExecuteDefend::ExecuteSeekCover()`:
```cpp
// Replace FindNearestCover() call with BTTask_FindCoverLocation
// This requires refactoring ExecuteSeekCover to use subtask
```

**Option B: Add EQS as Separate BT Task**
Add new node in Defend Subtree:
```
Defend Subtree
├─ [CheckTacticalAction: SeekCover] Find Cover Sequence
│  ├─ BTTask_FindCoverLocation (EQS)  ← NEW
│  └─ Move To (CoverLocation blackboard key)
└─ Execute Defend Task
```

**Option C: Use Both (Fallback)**
`BTTask_FindCoverLocation` already supports this:
```cpp
bUseLegacySearch = true    // Fallback to tag-based if EQS fails
CoverQuery = EQS_FindCover // Try EQS first
```

---

## 4. Testing & Debug

### Enable Debug Visualization
**Console Commands:**
```
ai.debug.eqs.show          // Show EQS queries
ai.debug.nav.show          // Show NavMesh
showdebug ai               // Show AI debug info
```

**BT Task Settings:**
- `BTTask_FindCoverLocation::bDrawDebug = true`
- `BTTask_ExecuteDefend::bDrawDebugInfo = true`

**Visual Indicators:**
- Yellow spheres: EQS candidate points
- Green sphere: Selected cover
- Cyan line: Path to cover
- Red lines: Enemy threats
- Green circle: Defensive radius

### Verify Integration
1. PIE with 2+ agents (Red vs Blue team)
2. Issue "Defend" command to follower
3. Wait for RL policy to select "SeekCover"
4. Check debug visualization for EQS query
5. Verify agent moves to selected cover

---

## 5. Performance Tuning

### EQS Parameters
```
GridSpacing: ↑ = Fewer samples, faster (less precision)
SearchRadius: ↓ = Fewer samples, faster
bCheckNavigability: Expensive - disable if not needed
bCheckLineOfSight: Moderate cost - adjust weight if slow
```

### Expected Performance
- EQS Query: 5-15ms (depends on grid size)
- Tag-based: 1-3ms (simple distance check)
- Recommendation: Use EQS for strategic cover, tag-based for quick reactions

---

## 6. Team System Integration

### Enemy Context for EQS
`EnvQueryContext_CoverEnemies` automatically fetches enemy positions from:
- `UTeamLeaderComponent::GetKnownEnemies()`
- Links follower → team leader → known enemies

**Flow:**
```
1. Agent perception detects enemy
2. Signal event to Team Leader (BTTask_SignalEventToLeader)
3. Team Leader registers enemy
4. EQS query uses EnvQueryContext_CoverEnemies
5. Cover test evaluates positions against enemy locations
```

**No manual setup required** - context automatically queries team system.

---

## 7. Simulation Manager GameMode

### Setup (Required for Team System)

**Step 1: Set GameMode**
- **World Settings** → **GameMode Override** → `SimulationManagerGameMode`

**Step 2: Register Teams (Blueprint or C++)**

Blueprint (BeginPlay):
```
Get Game Mode → Cast to SimulationManagerGameMode
├─ Register Team (ID: 0, Leader: RedLeaderComp, Name: "Red", Color: Red)
└─ Register Team (ID: 1, Leader: BlueLeaderComp, Name: "Blue", Color: Blue)
```

C++:
```cpp
ASimulationManagerGameMode* SimManager = Cast<ASimulationManagerGameMode>(
    GetWorld()->GetAuthGameMode());
if (SimManager) {
    SimManager->RegisterTeamMember(TeamID, this);
    TArray<AActor*> Enemies = SimManager->GetEnemyActors(TeamID);
}
```

**Step 3: Set Enemy Relationships**
```
SimManager->SetMutualEnemies(0, 1);  // Red vs Blue
```

---

## 8. Common Issues

### EQS returns no results
- Verify "Cover" tag on static meshes
- Check SearchRadius is sufficient
- Ensure NavMesh coverage
- Disable `bCheckNavigability` to test

### Cover ignores enemies
- Verify Team Leader has registered enemies (`GetKnownEnemies()`)
- Check `EnemyContext` is set to `EnvQueryContext_CoverEnemies`
- Ensure weights are non-zero

### Agents don't recognize enemies
- Verify enemy teams set via `SetMutualEnemies()`
- Check `GetEnemyActors()` returns expected results
- Debug with `AreActorsEnemies()`

---

## 9. Extension Examples

### Custom EQS Test
```cpp
UCLASS()
class UEnvQueryTest_CoverHeight : public UEnvQueryTest {
    virtual void RunTest(FEnvQueryInstance& QueryInstance) const override;
};
```

### Custom Context
```cpp
UCLASS()
class UEnvQueryContext_TeamObjective : public UEnvQueryContext {
    virtual void ProvideContext(FEnvQueryInstance& QueryInstance,
                                FEnvQueryContextData& ContextData) const override;
};
```

---

## Summary

**Current Status:**
- ✅ EQS C++ classes implemented
- ✅ Tag-based cover finding active (BTTask_ExecuteDefend)
- ✅ EQS-based cover finding available (BTTask_FindCoverLocation)
- 📋 Editor setup required: Create EQS asset, tag cover objects

**Quick Start:**
1. Tag cover objects with "Cover"
2. Test with tag-based method (works immediately)
3. Create EQS_FindCover asset for advanced evaluation
4. Integrate via Option A, B, or C (see Section 3)

**Files:**
- Generator: `EnvQueryGenerator_CoverPoints.h`
- Test: `EnvQueryTest_CoverQuality.h`
- Context: `EnvQueryContext_CoverEnemies.h`
- BT Task: `BTTask_FindCoverLocation.h`
- Execution: `BTTask_ExecuteDefend.cpp:290-346`
