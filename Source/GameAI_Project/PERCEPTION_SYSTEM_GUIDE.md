# AI Perception System Implementation Guide

**Problem:** Agents cannot detect enemies automatically, causing:
1. Empty enemy observations (19 enemy features = 0.0)
2. RL policy selecting movement actions instead of defensive actions
3. Defense branch decorators always failing
4. `TeamLeaderComponent::KnownEnemies` remaining empty

**Solution:** Implement AI Perception System to automatically detect and track enemies.

---

## Architecture Overview

```
AI Perception Component (per agent)
    ↓
Detect enemy (by sight/hearing/damage)
    ↓
Filter by TeamID (check if hostile)
    ↓
Register with Team Leader
    ↓
Update Observation (enemy features)
    ↓
RL Policy selects defensive action
    ↓
Defense branch executes
```

---

## Core Components to Implement

### 1. AI Perception Component Setup

**Location:** Attach to each follower agent actor (Blueprint or C++)

**Required Senses:**
- **AI Sight** - Primary enemy detection
  - Sight radius: 3000-5000 units
  - Lose sight radius: 5000-7000 units (wider than detection)
  - Peripheral vision angle: 90-120 degrees
  - Auto-success range: 500-1000 units (always detect close enemies)
  - Detection by affiliation: Detect enemies only

- **AI Hearing** (Optional) - Detect gunfire/footsteps
  - Hearing range: 2000-3000 units

- **AI Damage** (Optional) - Detect attackers instantly
  - Always enabled

**Configuration:**
- Dominant sense: AI Sight
- Update interval: 0.1-0.2 seconds (balance performance vs responsiveness)
- Forget time: 5-10 seconds (remember last known position)

### 2. Team Affiliation System

**Purpose:** Determine which actors are enemies based on TeamID

**Implementation Options:**

**Option A: Generic Team Agent Interface (Recommended)**
- Create `IGenericTeamAgentInterface` implementation
- Each agent implements `GetGenericTeamId()` → returns TeamID from instance variable
- AI Perception uses team affiliation for filtering
- Affiliation types:
  - Friendly: Same TeamID
  - Hostile: Different TeamID
  - Neutral: TeamID = -1 or 0

**Option B: Manual Filtering**
- Perception detects all actors
- OnTargetPerceptionUpdated callback manually filters by TeamID
- More control but more complex

**TeamID Source:**
- Instance variable on agent actor (editable per instance)
- Set in Blueprint (e.g., BP_FollowerAgent → TeamID = 1)
- Used by perception to determine hostility

### 3. Perception Event Handlers

**Required Callbacks:**

**A. OnTargetPerceptionUpdated**
- Triggered when actor enters/exits perception
- Parameters: Actor, stimulus (sight/hearing/damage)
- Logic:
  1. Get detected actor's TeamID
  2. Compare with own TeamID
  3. If hostile (different TeamID):
     - Register with Team Leader
     - Signal EnemySpotted event (priority 7)
     - Update local enemy list
  4. If friendly:
     - Ignore or update ally tracking

**B. OnPerceptionUpdated (Optional)**
- Batch update for all perceived actors
- More efficient for multiple simultaneous detections

**C. OnTargetPerceptionForgotten**
- Triggered when actor forgotten (after forget time)
- Logic:
  1. Check if enemy is dead or just out of sight
  2. If dead: Unregister from Team Leader
  3. If lost sight: Keep in Team Leader's known enemies (last known position)

### 4. Integration with Team Leader

**Automatic Enemy Registration:**
When enemy detected:
1. Get `TeamLeaderComponent` reference
2. Call `TeamLeader->RegisterEnemy(DetectedActor)`
3. Team Leader adds to `KnownEnemies` TSet
4. All followers can query `TeamLeader->GetKnownEnemies()`

**Enemy Elimination:**
When enemy killed:
1. Detect via health component death event
2. Call `TeamLeader->UnregisterEnemy(DeadActor)`
3. Team Leader increments `TotalEnemiesEliminated`
4. Triggers `EnemyEliminated` strategic event (priority 6)

**Shared Enemy Knowledge:**
- All team followers share enemy information via Team Leader
- Follower A detects enemy → Team Leader registers
- Follower B queries Team Leader → receives enemy list
- Enables coordinated tactics without direct perception

### 5. Integration with Observation System

**Current Problem:**
`ObservationElement::Build()` queries enemies, but if none registered, features 23-41 = 0.0

**Solution:**
After perception registers enemies:
1. `FollowerAgentComponent::GetLocalObservation()` called
2. Queries `TeamLeader->GetKnownEnemies()`
3. Selects 3 nearest/most relevant enemies
4. Populates enemy features (distance, angle, health, etc.)
5. RL network receives meaningful enemy data
6. Selects defensive actions (DefensiveHold, SeekCover, SuppressiveFire)

**Enemy Feature Updates:**
- Enemy 1-3 distance, angle, health, threat level (19 features)
- Enemy count (1 feature)
- All updated every observation refresh (0.1-0.2s)

### 6. Strategic Event Signaling

**New Events to Implement:**

**A. EnemySpotted (Priority 7)**
- Trigger: First detection of new enemy
- Frequency: Once per unique enemy
- Action: Team Leader runs MCTS (async)
- Result: Issues strategic commands (Assault/Defend/Retreat)

**B. EnemyLost (Priority 4)**
- Trigger: Enemy forgotten (out of sight for N seconds)
- Frequency: Once when forget time expires
- Action: May or may not trigger MCTS (below threshold)

**C. UnderFire (Priority 8)**
- Trigger: AI Damage sense activated
- Frequency: Immediate
- Action: Always triggers MCTS
- Result: Emergency tactical adjustment

**Implementation:**
- BT Service: `BTService_MonitorPerception`
- Checks perception component every tick
- Calls `FollowerComp->SignalEventToLeader()` on state changes

---

## Implementation Phases

### Phase 1: Basic Sight Detection (MVP)
**Goal:** Detect enemies by sight only

1. Add AI Perception Component to follower actors
2. Configure AI Sight sense (3000 unit radius, 90° angle)
3. Implement Generic Team Agent interface
4. Set TeamID instance variables on all agents
5. Test: Two agents with different TeamIDs should detect each other

**Success Criteria:**
- Agents detect enemies within sight radius
- `OnTargetPerceptionUpdated` callback fires
- Correct team affiliation filtering

### Phase 2: Team Leader Integration
**Goal:** Share enemy information across team

1. Create perception event handler (C++ or Blueprint)
2. On detection: Call `TeamLeader->RegisterEnemy()`
3. On forget/death: Call `TeamLeader->UnregisterEnemy()`
4. Test: One follower detects enemy → all team members aware

**Success Criteria:**
- Team Leader's `KnownEnemies` TSet populated
- `GetKnownEnemies()` returns detected enemies
- Debug drawing shows enemy indicators

### Phase 3: Observation Integration
**Goal:** Populate enemy features in observations

1. Verify `ObservationElement::Build()` uses Team Leader enemies
2. Test observation features 23-41 are non-zero
3. Verify RL policy receives enemy data
4. Test: RL selects defensive actions when enemies present

**Success Criteria:**
- Observation logs show enemy features (distance, angle, health)
- RL policy selects `DefensiveHold`, `SeekCover`, or `SuppressiveFire`
- Defense branch decorators pass
- Defense tasks execute

### Phase 4: Strategic Event System
**Goal:** Trigger MCTS on enemy detection

1. Signal `EnemySpotted` event to Team Leader (priority 7)
2. Team Leader runs async MCTS
3. Issues updated commands to followers
4. Test: Enemy detection causes tactical adjustment

**Success Criteria:**
- MCTS runs when enemy spotted
- Strategic commands updated
- FSM transitions to new state
- Coordinated team response

### Phase 5: Advanced Senses (Optional)
**Goal:** Add hearing and damage detection

1. Enable AI Hearing sense (footsteps, gunfire)
2. Enable AI Damage sense (instant attacker detection)
3. Implement stimulus-specific responses
4. Test: Damage from behind triggers immediate response

**Success Criteria:**
- Hearing detects off-screen enemies
- Damage sense triggers `UnderFire` event
- Faster reaction to ambushes

---

## Blueprint vs C++ Considerations

### Blueprint Implementation (Faster, Simpler)
**Pros:**
- No compilation time
- Visual debugging
- Easy to test/iterate
- Good for prototyping

**Approach:**
1. Add AI Perception Component to `BP_FollowerAgent`
2. Bind `OnTargetPerceptionUpdated` event
3. Use Blueprint nodes to filter by TeamID
4. Call `Register Enemy` on Team Leader Component

**Cons:**
- Performance overhead (minor for small teams)
- Less type safety

### C++ Implementation (Production-Ready)
**Pros:**
- Better performance
- Type safety
- Easier to maintain at scale
- Integrates with existing C++ systems

**Approach:**
1. Create `UAIPerceptionComponent` subclass or use base class
2. Override `HandleSenseDetection()` or bind delegates in `BeginPlay()`
3. Implement team filtering logic in C++
4. Call Team Leader functions directly

**Cons:**
- Compilation time
- Requires C++ knowledge

**Recommendation:** Start with Blueprint for Phase 1-2, migrate to C++ for Phase 3+ if performance issues arise.

---

## Configuration Best Practices

### Performance Optimization
1. **Update interval**: 0.2s for followers, 0.1s for leader
2. **Sight radius**: Balance detection vs performance
   - Small maps: 3000 units
   - Large maps: 5000 units
3. **Max simultaneous enemies**: Limit to 5-10 tracked enemies per team
4. **LOD system**: Reduce perception frequency for distant agents

### Debugging Tools
1. **AI Debug drawing**: `ShowDebug AI` console command
2. **Perception debug**: Shows sight cones, detected actors
3. **Team Leader debug**: Enable `bEnableDebugDrawing` to visualize enemies
4. **BT debug**: Shows which decorators pass/fail in real-time

### Common Pitfalls
1. **Forget time too short**: Enemies disappear immediately when out of sight
2. **TeamID not set**: All agents default to 0, treated as same team
3. **Affiliation not configured**: Perception detects allies as enemies
4. **No line-of-sight check**: Detect enemies through walls (enable LOS test)
5. **Stimulus expiration**: Old stimuli expire, causing forget events

---

## Integration with Existing Systems

### Behavior Tree Changes
**No changes required** - BT already has:
- `BTService_QueryRLPolicyPeriodic` - Queries RL every N seconds
- `BTDecorator_CheckTacticalAction` - Checks tactical action type
- Execute tasks for defensive actions

**Only requirement:** Perception populates `KnownEnemies` → RL receives enemy data → Selects defensive actions → Decorators pass

### RL Policy Changes
**No changes required** - RL network already:
- Accepts 71 observation features (including 19 enemy features)
- Has 16 tactical actions (including defensive actions)
- Trained to respond to enemy presence

**Only requirement:** Enemy features populated → Network selects appropriate actions

### MCTS Changes (Optional)
**Current:** Event-driven MCTS runs on high-priority events
**Addition:** Add `EnemySpotted` as high-priority trigger

**Benefit:** Team Leader immediately re-evaluates strategy when new enemies detected

---

## Testing Checklist

**Phase 1 - Detection:**
- [ ] Two agents with different TeamIDs placed in level
- [ ] Agent A within sight radius of Agent B
- [ ] `OnTargetPerceptionUpdated` fires for both agents
- [ ] Correct team filtering (detects enemy, ignores ally)

**Phase 2 - Team Integration:**
- [ ] Enemy detected by one follower
- [ ] `TeamLeader->KnownEnemies` contains enemy
- [ ] Other followers can query enemy list
- [ ] Debug visualization shows red sphere on enemy

**Phase 3 - Observation:**
- [ ] Observation logs show enemy distance/angle/health
- [ ] RL policy receives non-zero enemy features
- [ ] RL selects defensive actions (not movement)
- [ ] Defense decorators pass

**Phase 4 - Strategy:**
- [ ] Enemy detection triggers MCTS
- [ ] Strategic commands updated (log shows command change)
- [ ] FSM transitions to new state
- [ ] Coordinated team behavior

**Phase 5 - Edge Cases:**
- [ ] Enemy killed → unregistered from Team Leader
- [ ] Enemy out of sight → still tracked (last known position)
- [ ] Multiple enemies → correct prioritization (nearest/most threatening)
- [ ] Friendly fire → no self-detection

---

## Next Steps After Implementation

1. **Tune perception parameters** - Adjust radii, angles, update rates
2. **Train RL policy with enemy data** - Current policy may not respond optimally
3. **Add memory system** - Remember last known enemy positions
4. **Implement prediction** - Predict enemy movement/tactics
5. **Add communication** - Followers share enemy intel verbally (optional)

---

## File Locations for Implementation

**Blueprint:**
- `Content/Game/Blueprints/Actor/Characters/BP_FollowerAgent.uasset` - Add perception component
- `Content/Game/Blueprints/AI/BB_FollowerAgent.uasset` - Verify blackboard keys

**C++ (if needed):**
- `Source/GameAI_Project/Public/Team/PerceptionHandlerComponent.h` (new)
- `Source/GameAI_Project/Private/Team/PerceptionHandlerComponent.cpp` (new)
- `Source/GameAI_Project/Public/BehaviorTree/Services/BTService_MonitorPerception.h` (new)

**Existing files to reference:**
- `Team/TeamLeaderComponent.h/cpp` - `RegisterEnemy()`, `UnregisterEnemy()`, `GetKnownEnemies()`
- `Team/FollowerAgentComponent.h/cpp` - `SignalEventToLeader()`
- `Observation/ObservationElement.cpp:178-269` - Enemy feature calculation

---

## Summary

**Root Cause:** No enemy detection → Empty observations → Wrong RL actions → Defense branch fails

**Solution:** Implement AI Perception System → Detect enemies by TeamID → Register with Team Leader → Populate observations → RL selects defense → Defense branch executes

**Estimated Effort:**
- Phase 1 (Blueprint): 1-2 hours
- Phase 2 (Integration): 2-3 hours
- Phase 3 (Testing): 1-2 hours
- Phase 4 (Events): 1-2 hours
- **Total:** 5-9 hours for full implementation

**Priority:** **CRITICAL** - System non-functional without enemy detection
