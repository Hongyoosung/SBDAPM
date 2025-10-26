# Flee Behavior Implementation Guide

**Project:** SBDAPM - State-Based Dynamic Action Planning Model
**Component:** Flee Behavior Tree Subtree & FleeState Logic
**Status:** C++ Implementation Complete - Editor Setup Required
**Date:** 2025-10-27

---

## 1. Settings in the Editor

### 1.1 Blackboard Asset Configuration

**File:** `Content/AI/BB_SBDAPM.uasset` (or your Blackboard asset name)

Open your Blackboard asset and ensure the following keys exist:

| Key Name | Type | Description |
|----------|------|-------------|
| `CurrentStrategy` | String | Current strategic state ("Flee", "Attack", "MoveTo", "Dead") |
| `CoverLocation` | Vector | Target cover position for flee behavior |
| `ThreatLevel` | Float | Current threat level (0.0 - 1.0) |
| `TargetEnemy` | Object (Actor) | Current target enemy (used by Attack state) |
| `Destination` | Vector | Movement destination (used by MoveTo state) |

**Steps:**
1. Open `Content/AI/BB_SBDAPM.uasset` in the Behavior Tree editor
2. Click "New Key" for any missing keys
3. Set the correct type for each key
4. Save the Blackboard asset

---

### 1.2 Behavior Tree Structure Setup

**File:** `Content/AI/BT_SBDAPM.uasset` (or your Behavior Tree asset name)

#### **Root Structure:**

```
Root (Selector)
├─ [Decorator: CheckStrategy == "Dead"] DeadBehavior
│  └─ Task: Wait (indefinite)
│
├─ [Decorator: CheckStrategy == "Flee"] FleeBehavior ← NEW SUBTREE
│  └─ (See detailed structure below)
│
├─ [Decorator: CheckStrategy == "Attack"] AttackBehavior
│  └─ (Existing attack logic)
│
└─ [Decorator: CheckStrategy == "MoveTo"] MoveToBehavior
   └─ (Existing movement logic)
```

#### **Flee Subtree Detailed Structure:**

```
[Decorator: CheckStrategy == "Flee"] FleeBehavior
├─ [Service: UpdateObservation] (Tick every 0.5s)
│
└─ Sequence (Flee execution)
   ├─ Selector (Try cover first, fallback to evasive)
   │  │
   │  ├─ Sequence (Cover-based flee)
   │  │  ├─ Task: FindCoverLocation
   │  │  │  - SearchRadius: 1500.0
   │  │  │  - CoverLocationKey: "CoverLocation"
   │  │  │  - CoverTag: "Cover"
   │  │  │  - DrawDebug: true (for testing)
   │  │  │
   │  │  ├─ Task: MoveTo
   │  │  │  - Blackboard Key: CoverLocation
   │  │  │  - Acceptable Radius: 100.0
   │  │  │
   │  │  └─ Task: Wait
   │  │     - Wait Time: 1.0 seconds
   │  │
   │  └─ Task: EvasiveMovement (fallback if FindCover fails)
   │     - Duration: 2.0 seconds
   │     - MoveDistance: 300.0
   │     - MoveInterval: 0.5 seconds
   │     - DrawDebug: true (for testing)
   │
   └─ Task: Wait
      - Wait Time: 0.5 seconds (before re-evaluating)
```

#### **Step-by-Step BT Setup:**

1. **Open Behavior Tree Editor:**
   - Navigate to `Content/AI/BT_SBDAPM.uasset`
   - Double-click to open

2. **Add Flee Subtree Branch:**
   - Right-click on the Root Selector
   - Add Composite → Sequence (name it "FleeBehavior")

3. **Add CheckStrategy Decorator:**
   - Right-click on the FleeBehavior sequence
   - Add Decorator → "Check Strategy"
   - In Details panel:
     - Required Strategy: "Flee"
     - Strategy Key Name: "CurrentStrategy"
     - Enable Debug Log: true (for testing)

4. **Add UpdateObservation Service:**
   - Right-click on the FleeBehavior sequence
   - Add Service → "Update Observation"
   - In Details panel:
     - Interval: 0.5 seconds
     - Random Deviation: 0.1 seconds

5. **Add Cover/Evasive Selector:**
   - Drag from FleeBehavior sequence
   - Add Composite → Selector (name it "CoverOrEvasive")

6. **Add Cover Sequence:**
   - Drag from Selector
   - Add Composite → Sequence (name it "TryCover")
   - Add these tasks in order:

   **a. FindCoverLocation Task:**
   - Add Task → "Find Cover Location"
   - Settings:
     - Search Radius: 1500.0
     - Cover Location Key: "CoverLocation"
     - Cover Tag: "Cover"
     - Draw Debug: ✓ (checked for testing)

   **b. MoveTo Task:**
   - Add Task → "Move To" (built-in UE task)
   - Settings:
     - Blackboard Key: CoverLocation
     - Acceptable Radius: 100.0
     - Use Pathfinding: ✓
     - Allow Strafe: ✓

   **c. Wait Task:**
   - Add Task → "Wait" (built-in UE task)
   - Settings:
     - Wait Time: 1.0 seconds

7. **Add EvasiveMovement Fallback:**
   - Drag from Selector (same level as TryCover sequence)
   - Add Task → "Evasive Movement"
   - Settings:
     - Duration: 2.0 seconds
     - Move Distance: 300.0
     - Move Interval: 0.5 seconds
     - Draw Debug: ✓ (checked for testing)

8. **Add Re-evaluation Wait:**
   - Drag from the main FleeBehavior sequence (after Selector)
   - Add Task → "Wait"
   - Settings:
     - Wait Time: 0.5 seconds

9. **Save Behavior Tree**

---

### 1.3 Cover Actor Setup

The flee behavior requires actors tagged with "Cover" in your level.

#### **Option A: Tag Existing Static Meshes**

1. Select any Static Mesh Actor in your level (walls, crates, rocks, etc.)
2. In Details panel, scroll to "Tags"
3. Click "+" to add a new tag
4. Enter: `Cover`
5. Repeat for multiple cover locations throughout the level

**Recommended:** Tag 5-10 cover objects within 1500cm of where agents spawn for testing.

#### **Option B: Create Dedicated Cover Actors**

1. Create new Blueprint: `BP_CoverPoint`
   - Parent Class: Actor
   - Add a Billboard component for visibility in editor
   - Add a Box component to visualize cover area

2. In the Blueprint:
   - Add Tag "Cover" in the Class Defaults
   - Optional: Add a static mesh (low wall, barrier, etc.)

3. Place multiple `BP_CoverPoint` actors in your level

#### **Verification:**

1. In World Outliner, search: `tag:Cover`
2. You should see all tagged actors highlighted
3. Ensure they're within 1500cm of your AI agent spawn points

---

### 1.4 AI Controller & StateMachine Configuration

#### **AI Controller Blueprint Setup:**

**File:** `BP_SBDAPMController` (or your AI Controller Blueprint)

1. Open your AI Controller Blueprint
2. Ensure it has a `SBDAPMController` C++ component (or uses it as parent class)
3. In Event BeginPlay:
   - Verify Behavior Tree is assigned: `BT_SBDAPM`
   - Verify Blackboard is assigned: `BB_SBDAPM`
   - Call `Run Behavior Tree` with your BT asset

4. Verify StateMachine component exists:
   - Components panel → should have `StateMachine` component
   - If missing, add it: Add Component → "State Machine"

#### **StateMachine Initial State:**

1. In your AI Controller or Pawn Blueprint
2. In Event BeginPlay, initialize states:
   ```blueprint
   Get State Machine Component
   → Initialize States (create MoveToState, AttackState, FleeState, DeadState)
   → Set Initial State (typically MoveToState)
   ```

---

### 1.5 Test Level Setup

Create a simple test scenario:

#### **Recommended Test Setup:**

1. **Floor:** 5000x5000 unit plane with NavMesh
2. **Cover Objects:** 5-6 Static Meshes tagged "Cover" (100-1500cm from spawn)
3. **AI Agent:** Your SBDAPM AI character
4. **Test Trigger:** Blueprint or console command to:
   - Set agent health to 30%
   - Spawn 3-5 enemy actors nearby
   - Force transition to FleeState

#### **Quick Test Trigger Blueprint:**

Create a Blueprint (F key trigger):
```blueprint
Event: F Key Pressed
→ Get Player Pawn
→ Get AI Controller
→ Get State Machine Component
→ Update Agent State:
   - Health: 25.0
   - Stamina: 50.0
→ Update Enemy Info:
   - Visible Count: 5
   - Nearby Enemies: [create 3-5 mock enemy observations]
→ Change State: FleeState
```

---

### 1.6 Compilation & Hot Reload

Before testing in editor:

1. **Compile C++ Code:**
   ```bash
   # In Visual Studio or Rider
   Build → Build Solution (Ctrl+Shift+B)
   ```

2. **Wait for successful compilation** (check Output window)

3. **Hot Reload in Editor:**
   - If editor is open: It should auto-reload
   - If not: Close and reopen Unreal Editor

4. **Verify New Classes Appear:**
   - In Content Browser → View Options → Show C++ Classes
   - Search for: `BTTask_FindCoverLocation`, `BTTask_EvasiveMovement`
   - They should appear in the list

---

## 2. Things to Test in the Editor

### 2.1 Pre-Flight Checks

Before running gameplay tests, verify:

- [ ] **Compilation:** No errors in Output Log
- [ ] **Blackboard Keys:** All required keys exist
- [ ] **BT Structure:** Flee subtree created with correct decorators
- [ ] **Cover Actors:** At least 3-5 actors tagged "Cover" in level
- [ ] **AI Controller:** Assigned BT and Blackboard assets
- [ ] **NavMesh:** Green NavMesh covers test area (Press 'P' to visualize)

### 2.2 Static Validation Tests

#### **Test 1: Behavior Tree Decorator Check**

**Purpose:** Verify BTDecorator_CheckStrategy correctly filters subtrees

**Steps:**
1. Open BT_SBDAPM in editor
2. Click on the Flee subtree's "Check Strategy" decorator
3. Details panel → Enable Debug Log: ✓
4. PIE (Play In Editor)
5. Open Output Log (Window → Developer Tools → Output Log)
6. Search for: "BTDecorator_CheckStrategy"
7. You should see logs showing strategy checks

**Expected:**
```
LogTemp: BTDecorator_CheckStrategy: Strategy MISMATCH - Current: 'MoveTo', Required: 'Flee'
```

**Pass Criteria:** Decorator correctly evaluates strategy match/mismatch

---

#### **Test 2: Cover Actor Detection**

**Purpose:** Verify FindCoverLocation can find tagged actors

**Steps:**
1. Place AI agent in level
2. Ensure 2-3 "Cover" tagged actors within 1500cm
3. PIE
4. Force agent into FleeState (use test trigger)
5. Watch for "FindCoverLocation" logs in Output Log

**Expected:**
```
LogTemp: BTTask_FindCoverLocation: Found cover at (X=1234.5, Y=567.8, Z=0.0)
LogTemp: BTTask_FindCoverLocation: Selected cover at distance 1234.5 (Score: 265.5)
```

**Pass Criteria:** Task finds at least one cover location

**Failure Debug:**
- If "No actors with tag 'Cover' found": Check actor tags
- If "No cover found within radius": Increase SearchRadius or move cover closer

---

#### **Test 3: Blackboard Updates**

**Purpose:** Verify FleeState writes to Blackboard correctly

**Steps:**
1. PIE with Behavior Tree debugger open (Window → Gameplay Debugger)
2. Select your AI agent
3. Force FleeState entry
4. In Gameplay Debugger, expand Blackboard view
5. Check values update in real-time

**Expected Blackboard Values:**
- `CurrentStrategy`: "Flee" (String)
- `CoverLocation`: Valid vector (not 0,0,0) when cover available
- `ThreatLevel`: 0.5-1.0 (Float) when enemies nearby

**Pass Criteria:** All three keys update correctly when FleeState is active

---

### 2.3 Dynamic Behavior Tests

#### **Test 4: Cover-Based Flee**

**Purpose:** Verify agent moves to cover when available

**Scenario Setup:**
- Agent health: 25%
- Visible enemies: 5
- Cover within 1000cm
- FleeState active

**Steps:**
1. PIE
2. Trigger flee scenario
3. Enable BT debug visualization (Apostrophe key ' or Gameplay Debugger)
4. Observe agent behavior

**Expected Behavior:**
1. FleeState enters (log: "Entered FleeState")
2. MCTS initializes (log: "FleeState: Initialized MCTS with 3 possible actions")
3. FindCoverLocation succeeds (green debug sphere appears at cover)
4. Agent pathfinds to cover (blue line shows path)
5. Agent reaches cover and waits 1 second
6. Sequence repeats (re-evaluation every 0.5s)

**Visual Indicators (if DrawDebug = true):**
- **Green Sphere (large):** Selected cover location
- **Yellow Spheres:** Candidate cover locations
- **Red Spheres:** Out-of-range cover
- **Cyan Line:** Path from agent to cover

**Pass Criteria:** Agent successfully reaches cover within 5 seconds

---

#### **Test 5: Evasive Movement Fallback**

**Purpose:** Verify zigzag movement when no cover available

**Scenario Setup:**
- Agent health: 25%
- Visible enemies: 5
- **NO cover within 1500cm** (remove/disable cover actors)
- FleeState active

**Steps:**
1. Remove all "Cover" tagged actors from level
2. PIE
3. Trigger flee scenario
4. Observe agent behavior

**Expected Behavior:**
1. FleeState enters
2. FindCoverLocation **fails** (log: "No cover found within radius 1500.0")
3. Selector falls back to EvasiveMovement task
4. Agent performs zigzag movement for 2 seconds:
   - Changes direction every 0.5 seconds
   - Moves 300cm in random directions
   - Creates unpredictable path

**Visual Indicators (if DrawDebug = true):**
- **Yellow Lines:** Zigzag movement path
- **Orange Spheres:** Movement target points
- **Cyan Arrows:** Direction changes

**Pass Criteria:** Agent performs erratic movement for ~2 seconds, then repeats

---

#### **Test 6: MCTS Decision Making**

**Purpose:** Verify MCTS runs and selects flee actions

**Steps:**
1. PIE
2. Trigger FleeState
3. Monitor Output Log for MCTS logs

**Expected Logs:**
```
LogTemp: Entered FleeState
LogTemp: FleeState: Initialized MCTS with 3 possible actions
LogTemp: FleeState: Generated 3 possible flee actions
LogTemp: RunMCTS Start - CurrentNode: 0x..., TreeDepth: 1
LogTemp: Expand: Create New Node
LogTemp: Selected Child with UCT Value: 1.234
LogTemp: FleeAction: Sprint to Cover selected by MCTS
LogTemp: Backpropagate: Update Node - VisitCount: 1, TotalReward: 123.45
```

**Pass Criteria:** MCTS expands tree, selects actions, backpropagates rewards

---

#### **Test 7: Reward Calculation (Flee-Specific)**

**Purpose:** Verify flee-specific rewards are calculated

**Scenario:** Agent with low health (<40%) and multiple enemies (>2)

**Steps:**
1. PIE
2. Trigger flee scenario
3. Search Output Log for: "MCTS Flee Reward"

**Expected Logs:**
```
LogTemp: MCTS Flee Reward: Total=215.3 (Cover=100.0, DistFromEnemy=45.2, Health=12.5, Stamina=0.0, CoverDist=37.6)
```

**Pass Criteria:** Flee-specific reward breakdown appears (not default reward)

**Comparison Test:**
- In non-flee scenario (health >40%, enemies <2):
```
LogTemp: MCTS Default Reward: Total=123.4 (Distance=50.0, Health=70.0, Enemy=-20.0)
```

---

#### **Test 8: State Transition (Exit FleeState)**

**Purpose:** Verify agent exits FleeState when safe

**Exit Conditions (from NEXT_STEPS3.md):**
- No enemies within 2000 units, OR
- Health restored > 40%, OR
- Reached cover AND Health > 25%

**Steps:**
1. PIE with agent in FleeState
2. Simulate safety:
   - Remove enemies (or set VisibleEnemyCount = 0)
   - OR increase agent health to 45%
3. Observe state transition

**Expected:**
```
LogTemp: Exited FleeState - MCTS backpropagation complete
LogTemp: Entered MoveToState (or AttackState)
```

**Pass Criteria:** Agent transitions out of FleeState when conditions met

**Note:** State transition logic must be implemented in your StateMachine TickComponent or state update logic. This may require additional implementation.

---

### 2.4 Edge Case Tests

#### **Test 9: No NavMesh Coverage**

**Purpose:** Verify graceful handling when target location has no navmesh

**Steps:**
1. Place cover actor in area WITHOUT navmesh (outside walkable area)
2. PIE and trigger FleeState

**Expected:**
- FindCoverLocation may still find the actor
- MoveTo task should fail or navigate to nearest valid point
- Agent should not get stuck

**Pass Criteria:** No crashes, agent continues attempting flee behavior

---

#### **Test 10: Rapid State Changes**

**Purpose:** Verify system handles quick Enter/Exit cycles

**Steps:**
1. Create Blueprint that rapidly toggles FleeState:
   ```blueprint
   Event Tick
   → Every 1 second:
      → Change State to FleeState
      → Wait 0.5s
      → Change State to MoveToState
   ```
2. PIE and observe

**Expected:**
- MCTS initializes/backpropagates cleanly
- No memory leaks or null pointer errors
- Logs show proper Enter/Exit sequence

**Pass Criteria:** No crashes, clean state transitions in logs

---

### 2.5 Performance Tests

#### **Test 11: Frame Rate Impact**

**Purpose:** Measure performance impact of MCTS + BT

**Steps:**
1. PIE
2. Open console (~)
3. Type: `stat fps`
4. Trigger FleeState with multiple AI agents (5-10)
5. Monitor FPS

**Expected:**
- FPS > 30 with 5 agents fleeing simultaneously
- No severe frame drops during MCTS RunMCTS calls

**Pass Criteria:** Playable frame rate maintained

**Optimization Note:** If FPS < 30, consider:
- Reducing MCTS iterations
- Running MCTS on background thread (async)
- Reducing tree depth limit

---

#### **Test 12: Memory Stability**

**Purpose:** Verify no memory leaks during extended flee behavior

**Steps:**
1. PIE
2. Console: `stat memory`
3. Let AI flee for 5+ minutes
4. Monitor memory usage

**Pass Criteria:** Memory usage remains stable (no continuous growth)

---

### 2.6 Integration Tests

#### **Test 13: Full State Machine Cycle**

**Purpose:** Verify all states work together

**Scenario:** Complete AI lifecycle

**Steps:**
1. Start in MoveToState (agent moving to destination)
2. Spawn enemies nearby → should transition to AttackState
3. Damage agent to <30% health → should transition to FleeState
4. Remove enemies OR heal agent → should transition back

**Expected:**
```
LogTemp: Entered MoveToState
LogTemp: Exited MoveToState
LogTemp: Entered AttackState
LogTemp: Exited AttackState
LogTemp: Entered FleeState
LogTemp: Exited FleeState
LogTemp: Entered MoveToState
```

**Pass Criteria:** Smooth transitions through all states based on conditions

---

## 3. Next Steps If All Pass

### 3.1 Immediate Post-Testing Tasks

#### **A. Disable Debug Visualization**

Once testing is complete and behavior is validated:

1. **Behavior Tree Tasks:**
   - FindCoverLocation → DrawDebug: ✗ (unchecked)
   - EvasiveMovement → DrawDebug: ✗ (unchecked)

2. **Decorators:**
   - BTDecorator_CheckStrategy → Enable Debug Log: ✗ (unchecked)

3. **MCTS Logs:**
   - In MCTS.cpp, change log levels from `Warning` to `Verbose`:
   ```cpp
   UE_LOG(LogTemp, Verbose, TEXT("...")); // Instead of Warning
   ```
   - Or conditionally compile out with `#if UE_BUILD_DEBUG`

---

#### **B. Tune Parameters**

Based on testing observations, adjust:

| Parameter | Location | Tuning Guidance |
|-----------|----------|-----------------|
| **SearchRadius** | BTTask_FindCoverLocation | Increase if "no cover found" frequently; Decrease for performance |
| **Duration** | BTTask_EvasiveMovement | Increase for longer evasion (2-5s); Decrease for more responsive behavior |
| **MoveDistance** | BTTask_EvasiveMovement | Increase for wider zigzags (400-600cm); Decrease for tighter movement |
| **MoveInterval** | BTTask_EvasiveMovement | Increase for smoother paths (0.7-1.0s); Decrease for more erratic (0.3-0.5s) |
| **Flee Reward Weights** | MCTS.cpp line 233-266 | Adjust multipliers to prioritize cover vs. distance vs. health |

**Recommended Tuning Process:**
1. Create Blueprint exposed parameters (convert hardcoded values to UPROPERTYs)
2. Use Data Tables or Curves for difficulty scaling
3. A/B test different configurations

---

#### **C. Document Learnings**

Create a summary document:
- Edge cases discovered during testing
- Performance bottlenecks identified
- Parameter values that worked best
- Visual/audio feedback that enhanced player experience

---

### 3.2 Feature Enhancements

#### **Priority 1: Complete State Transition Logic**

**Current Gap:** NEXT_STEPS3.md specifies exit conditions, but they're not implemented in code yet.

**Implementation Required:**
- **File:** `Private/Core/StateMachine.cpp` (in TickComponent or state update)
- **Logic:**
  ```cpp
  void UStateMachine::TickComponent(float DeltaTime, ...)
  {
      UState* CurrentState = GetCurrentState();

      // FleeState exit conditions
      if (CurrentState->IsA<UFleeState>())
      {
          FObservationElement Obs = GetCurrentObservation();

          // Exit to MoveToState if safe
          if (Obs.VisibleEnemyCount == 0 ||
              Obs.Health > 40.0f ||
              (Obs.bHasCover && Obs.Health > 25.0f))
          {
              ChangeState(GetMoveToState());
              return;
          }
      }

      // Continue with state update...
      CurrentState->UpdateState(this, Reward, DeltaTime);
  }
  ```

**Testing:** Verify agent exits FleeState when conditions are met

---

#### **Priority 2: Enhance Cover Evaluation**

**Current Limitation:** FindCoverLocation only considers distance, not tactical quality.

**Enhancements:**
1. **Line-of-sight Check:**
   - Raycast from cover to enemies
   - Prefer cover that blocks enemy LOS

2. **Cover Scoring:**
   ```cpp
   float CoverScore =
       (SearchRadius - Distance) * 0.4f +          // Proximity
       (DistanceFromEnemies / 10.0f) * 0.4f +       // Safety
       (HasLineOfSightBlock ? 100.0f : 0.0f) * 0.2f; // Protection
   ```

3. **Escape Routes:**
   - Check if cover has multiple exit paths
   - Penalize "corner" positions

**Files to Modify:**
- `BTTask_FindCoverLocation.cpp::FindBestCover()`

---

#### **Priority 3: Fight While Retreating**

**Current Status:** FightWhileRetreatingAction exists but isn't executed by BT.

**Implementation:**
1. **Detect when action is selected:**
   - In FleeState.cpp, track which MCTS action was chosen
   - Set a Blackboard key: `FleeStrategy` = "FightWhileRetreating"

2. **Create BT subtree:**
   ```
   [Decorator: FleeStrategy == "FightWhileRetreating"]
   └─ Parallel
      ├─ MoveTo CoverLocation
      └─ Sequence (Fire periodically)
         ├─ Wait (random 0.5-1.5s)
         ├─ RotateToFaceBBEntry (TargetEnemy)
         ├─ ExecuteAttack
         └─ Loop
   ```

3. **Test:** Agent fires at enemies while moving to cover

---

#### **Priority 4: Advanced Evasive Movement**

**Current:** Simple zigzag with random angles

**Enhancements:**
1. **Enemy-Aware Evasion:**
   - Read enemy positions from observation
   - Move perpendicular to enemy fire direction
   - Use cover-to-cover dashing

2. **Stamina System Integration:**
   - Sprint when stamina > 50%
   - Walk when stamina < 20%
   - Manage stamina resource

3. **Animation Integration:**
   - Trigger dodge roll animation at direction changes
   - Play sprint animation during high-speed movement

**Files to Modify:**
- `BTTask_EvasiveMovement.cpp::TickTask()`

---

### 3.3 Polish & Production Readiness

#### **A. Animation State Machine Integration**

**Current:** Movement is purely positional (no animation sync)

**Implementation:**
1. **Create Animation Blueprint:**
   - Add Flee locomotion state
   - Blend trees for sprint, jog, combat-ready walk
   - Evasive dodge animations

2. **Sync with FleeState:**
   - Set animation parameters based on Blackboard
   - `IsFleeing` boolean
   - `FleeIntensity` float (0-1, based on threat level)

3. **Test:** Agent visually appears to flee with appropriate urgency

---

#### **B. Audio Feedback**

**Additions:**
1. **Flee Entry Sound:** Panic breathing, alert vocalization
2. **Evasive Movement:** Quick footsteps, gear jostling
3. **Cover Reached:** Relief exhale, "Safe!" callout

**Implementation:**
- Add Sound Cues to BT tasks
- Use Audio Components on AI pawn

---

#### **C. Player Feedback (UI)**

**For player-facing AI:**
1. **Status Indicator:** Above AI head, color-coded by state
   - Red = Attack
   - Yellow = MoveTo
   - Blue = Flee
   - Gray = Dead

2. **Threat Radar:** Minimap showing AI flee paths

3. **Debug Commands:**
   ```
   ai.debug.showstates 1
   ai.debug.showcover 1
   ai.debug.mcts.verbose 1
   ```

---

### 3.4 Multi-Agent Scenarios

#### **Flee Coordination:**

**Enhancement:** Multiple agents coordinate flee behavior

**Features:**
1. **Avoid Same Cover:**
   - Share cover reservation via GameMode or AIController manager
   - Each agent claims different cover location

2. **Suppressing Fire:**
   - Designate one agent to FightWhileRetreating
   - Others focus on reaching safety

3. **Leader-Follower:**
   - First agent to flee becomes "leader"
   - Others follow to same general area

**Implementation:**
- Create `AFleeCoordinator` actor
- Agents register with coordinator when entering FleeState
- Coordinator assigns cover and roles

---

### 3.5 Machine Learning / Training

#### **RL Integration (Advanced):**

**Goal:** Replace hardcoded MCTS rewards with learned policy

**Approach 1: Integrate Unreal LearningAgents Plugin**
1. Enable LearningAgents plugin (already in .uproject)
2. Create `UFleePolicy` class inheriting from `ULearningAgentsPolicy`
3. Define observation space (71 features)
4. Define action space (3 flee actions)
5. Train using Imitation Learning → PPO
6. Replace MCTS with neural network inference

**Approach 2: External RL Framework (Ray RLlib)**
1. Create Python interface to Unreal via sockets
2. Implement custom Gym environment
3. Train PPO agent offline
4. Export trained weights
5. Load weights in C++ for inference

**Files to Create:**
- `Source/.../FleePolicy.h/cpp`
- `Python/train_flee_policy.py`

**Benefits:**
- Agents learn optimal flee strategies from experience
- Adapts to player behavior
- Generalizes to new scenarios without manual tuning

---

### 3.6 Quality Assurance

#### **Automated Testing:**

**Create Unit Tests:**
- **File:** `Source/GameAI_Project/Tests/FleeStateTests.cpp`
- **Framework:** Unreal Automation Framework

**Test Cases:**
```cpp
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FFleeStateMCTSInitTest,
    "GameAI.FleeState.MCTS.Initialization",
    EAutomationTestFlags::ApplicationContextMask |
    EAutomationTestFlags::ProductFilter)

bool FFleeStateMCTSInitTest::RunTest(const FString& Parameters)
{
    UFleeState* FleeState = NewObject<UFleeState>();
    UStateMachine* MockStateMachine = NewObject<UStateMachine>();

    FleeState->EnterState(MockStateMachine);

    // Verify MCTS initialized
    TestNotNull("MCTS should be initialized", FleeState->MCTS);

    // Verify 3 possible actions
    TArray<UAction*> Actions = FleeState->GetPossibleActions();
    TestEqual("Should have 3 flee actions", Actions.Num(), 3);

    return true;
}
```

**Run Tests:**
- Session Frontend → Automation → Run "GameAI" tests

---

### 3.7 Documentation Updates

#### **Update CLAUDE.md:**

Add section:
```markdown
### FleeState Implementation (Completed)

**Status:** ✅ Complete (v1.0)

**Features:**
- MCTS-driven flee strategy selection
- Cover-based retreat (FindCoverLocation)
- Evasive zigzag movement fallback
- Flee-specific reward function

**Files:**
- States/FleeState.h/cpp
- Actions/FleeActions/*.h/cpp
- BehaviorTree/BTTask_FindCoverLocation.h/cpp
- BehaviorTree/BTTask_EvasiveMovement.h/cpp

**Testing:** See FLEE_IMPLEMENTATION_GUIDE.md

**Known Limitations:**
- Cover evaluation is distance-based only (no tactical scoring)
- State transition conditions not fully implemented
- FightWhileRetreating action not connected to BT execution
```

---

### 3.8 Advanced Research Topics

#### **Future Research Directions:**

1. **Hierarchical RL:**
   - High-level policy: Selects flee strategy
   - Low-level policy: Executes movement primitives
   - Reduces action space complexity

2. **Inverse RL:**
   - Learn reward function from player demonstrations
   - Capture expert flee behavior
   - Transfer to AI agents

3. **Multi-Agent RL:**
   - Agents learn cooperative flee tactics
   - Emergent behaviors (covering fire, staged retreat)

4. **Transfer Learning:**
   - Pre-train on simple scenarios
   - Fine-tune on complex combat situations
   - Domain randomization for robustness

5. **Explainable AI:**
   - Visualize MCTS tree in-game
   - Show decision rationale to designers
   - Debug tools for non-programmers

---

### 3.9 Production Deployment Checklist

Before shipping:

- [ ] **Remove Debug Logs:** Set MCTS logs to `Verbose` or compile out
- [ ] **Disable Debug Visualization:** Uncheck all DrawDebug flags
- [ ] **Profile Performance:** Ensure <1ms per agent for MCTS update
- [ ] **Test at Scale:** 20+ agents fleeing simultaneously
- [ ] **Memory Leak Test:** 1-hour flee scenario, monitor memory
- [ ] **Cross-Platform Test:** Windows, Console, Mobile (if applicable)
- [ ] **Save/Load Test:** Verify MCTS state persists (if needed)
- [ ] **Network Test:** Multiplayer flee behavior synchronization
- [ ] **Edge Case Coverage:** All error paths tested
- [ ] **Code Review:** Second engineer reviews all C++ changes
- [ ] **QA Sign-Off:** Testing team validates all requirements

---

## Summary

**Current Status:** ✅ C++ implementation complete, ready for editor setup

**Next Immediate Steps:**
1. Complete Behavior Tree setup in editor (Section 1.2)
2. Tag cover actors in test level (Section 1.3)
3. Run validation tests (Section 2)
4. If all tests pass → proceed with enhancements (Section 3)

**Estimated Time:**
- **Editor Setup:** 30-60 minutes
- **Testing:** 1-2 hours
- **Tuning & Polish:** 2-4 hours
- **Advanced Features:** 1-2 weeks per priority item

**Success Criteria:**
- Agent successfully flees to cover when health < 30%
- MCTS runs without errors and selects flee actions
- Evasive movement activates when no cover available
- State transitions work correctly
- Frame rate remains playable (>30 FPS)

---

**Document Version:** 1.0
**Last Updated:** 2025-10-27
**Maintained By:** Claude Code Assistant
