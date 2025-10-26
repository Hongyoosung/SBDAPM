# SBDAPM: Next Implementation Steps - Updated Prompts (Behavior Tree Architecture)

This document provides updated prompts reflecting the new **hybrid architecture** where Behavior Trees handle tactical execution while FSM/MCTS/RL control strategic decision-making.

**Architecture Change Summary:**
- **OLD:** FSM → MCTS → Simple Actions → Blueprint Events
- **NEW:** FSM → MCTS/RL → **Blackboard** → **Behavior Tree** → Tactical Execution (EQS, MoveTo, Custom Tasks)

---

## Phase 1: Foundation & Code Quality (Weeks 1-2)

### ✅ Step 2.1: Expand Observation Space [COMPLETED]

**Status:** Implemented successfully with 71 features.

**Completed:**
- Enhanced ObservationElement with 71 features (Agent, Combat, Perception, Enemies, Tactical, Temporal)
- Added FEnemyObservation struct and enums (ERaycastHitType, ETerrainType)
- Implemented ToFeatureVector() for neural network compatibility
- Updated StateMachine with observation management methods
- Fixed all MCTS.cpp compilation errors
- Full Blueprint exposure with proper categories

**Files Modified:**
- ✅ Public/Core/ObservationElement.h
- ✅ Private/Core/ObservationElement.cpp
- ✅ Public/Core/StateMachine.h
- ✅ Private/Core/StateMachine.cpp
- ✅ Private/AI/MCTS.cpp (field name updates)

---

### Step 2.2: Create Behavior Tree Integration Layer

**Prompt:**
```
I need to integrate Unreal Engine's Behavior Tree system with the SBDAPM strategic decision-making layer (FSM + MCTS).

The architecture should use:
- **Strategic Layer (FSM + MCTS):** Makes high-level decisions (attack vs flee vs move, target selection, strategy choice)
- **Tactical Layer (Behavior Tree):** Executes low-level actions (pathfinding, aiming, shooting, dodging, animations)
- **Communication:** Blackboard keys bridge the two layers

Requirements:

1. **Create AIController with Behavior Tree**
   - Public/AI/SBDAPMController.h
   - Private/AI/SBDAPMController.cpp
   - Override OnPossess() to start Behavior Tree
   - Access StateMachine component from controlled pawn
   - Run Behavior Tree with custom Blackboard

2. **Define Blackboard Asset**
   - Content/AI/BB_SBDAPM.uasset (create in Blueprint)
   - Keys needed:
     * CurrentStrategy (String): "MoveTo", "Attack", "Flee", "Dead"
     * TargetEnemy (Object): AActor reference
     * CoverLocation (Vector): Target cover position
     * Destination (Vector): Movement goal
     * ThreatLevel (Float): 0-1 danger assessment
     * bCanSeeEnemy (Bool)
     * LastObservationUpdate (Float): Timestamp

3. **Create Behavior Tree Structure**
   - Content/AI/BT_SBDAPM.uasset (create in Blueprint)
   - Root Selector node with decorators checking CurrentStrategy
   - Subtrees:
     * Dead Behavior (play death animation)
     * Flee Behavior (find cover, sprint to safety)
     * Attack Behavior (find position, aim, shoot)
     * MoveTo Behavior (pathfind to destination)

4. **Create Custom BT Decorator: CheckStrategy**
   - Public/BehaviorTree/BTDecorator_CheckStrategy.h
   - Private/BehaviorTree/BTDecorator_CheckStrategy.cpp
   - Checks if Blackboard.CurrentStrategy matches required strategy
   - Use for activating correct subtree

5. **Create Custom BT Service: UpdateObservation**
   - Public/BehaviorTree/BTService_UpdateObservation.h
   - Private/BehaviorTree/BTService_UpdateObservation.cpp
   - Runs every tick (configurable interval)
   - Gathers observation data:
     * Agent position, velocity, rotation, health
     * Performs 16-ray raycasts for environment perception
     * Scans for nearby enemies (top 5 closest)
     * Detects cover availability
     * Updates weapon/combat state
   - Calls StateMachine->UpdateObservation(FObservationElement)
   - Syncs key values to Blackboard (for BT tasks to use)

6. **Update FSM States to Write Blackboard**
   Modify existing states to control Behavior Tree:

   **MoveToState:**
   - In EnterState(): Set Blackboard.CurrentStrategy = "MoveTo"
   - In UpdateState(): Run MCTS to select movement strategy, update Blackboard.Destination

   **AttackState:**
   - In EnterState(): Set Blackboard.CurrentStrategy = "Attack"
   - In UpdateState(): Run MCTS to select target, update Blackboard.TargetEnemy

   **FleeState:**
   - In EnterState(): Set Blackboard.CurrentStrategy = "Flee"
   - In UpdateState(): Run MCTS to select cover, update Blackboard.CoverLocation

   **DeadState:**
   - In EnterState(): Set Blackboard.CurrentStrategy = "Dead", stop BT

7. **Create Helper Functions in StateMachine**
   Add methods for Blackboard communication:
   ```cpp
   // Get AI Controller (for Blackboard access)
   AAIController* GetAIController() const;

   // Get Blackboard Component
   UBlackboardComponent* GetBlackboard() const;

   // Set strategy on Blackboard
   void SetCurrentStrategy(const FString& Strategy);
   ```

Files to create:
- Public/AI/SBDAPMController.h
- Private/AI/SBDAPMController.cpp
- Public/BehaviorTree/BTDecorator_CheckStrategy.h
- Private/BehaviorTree/BTDecorator_CheckStrategy.cpp
- Public/BehaviorTree/BTService_UpdateObservation.h
- Private/BehaviorTree/BTService_UpdateObservation.cpp

Files to modify:
- Public/Core/StateMachine.h (add Blackboard helper methods)
- Private/Core/StateMachine.cpp
- Private/States/MoveToState.cpp (add Blackboard writes)
- Private/States/AttackState.cpp (add Blackboard writes)
- Private/States/FleeState.cpp (add Blackboard writes)
- Private/States/DeadState.cpp (add Blackboard writes)

Blueprint assets to create (in Unreal Editor after C++ compiles):
- Content/AI/BB_SBDAPM.uasset (Blackboard)
- Content/AI/BT_SBDAPM.uasset (Behavior Tree)

Please implement the C++ components with proper UPROPERTY declarations for Blueprint exposure. Include detailed comments explaining the strategic/tactical layer separation.
```

---

### Step 2.3: Implement Flee Behavior Tree Subtree & FleeState Logic

**Prompt:**
```
Now that we have the Behavior Tree integration layer, I need to complete the FleeState strategic logic and create the corresponding tactical Behavior Tree subtree.

Requirements:

1. **Complete FleeState Strategic Logic**
   FleeState should use MCTS to decide:
   - Which flee strategy? (Sprint to cover, Evasive movement, Fight while retreating)
   - Which cover location? (Use NearestCoverDistance and CoverDirection from observation)
   - When to stop fleeing? (Exit conditions)

   Update Private/States/FleeState.cpp:
   - EnterState(): Initialize MCTS, set Blackboard.CurrentStrategy = "Flee"
   - UpdateState():
     * Run MCTS with flee-specific actions (represented as strategy choices)
     * Evaluate cover locations using observation.NearbyEnemies data
     * Select best cover, write to Blackboard.CoverLocation
     * If no cover, choose evasive movement
   - ExitState(): Backpropagate rewards
   - GetPossibleActions(): Return flee strategies as "actions"
     * Action 1: "Sprint to nearest cover"
     * Action 2: "Sprint to safest cover (furthest from enemies)"
     * Action 3: "Evasive movement (zigzag)"
     * Action 4: "Fight while retreating"

   State Transition Logic:
   - Enter FleeState when:
     * Health < 20% AND VisibleEnemyCount > 3, OR
     * Health < 30% AND VisibleEnemyCount > 5, OR
     * Health < 40% AND bHasCover == false AND VisibleEnemyCount > 2

   - Exit FleeState when:
     * No enemies within 2000 units (safe distance reached), OR
     * Health restored > 40%, OR
     * Reached cover AND Health > 25%

2. **Create Custom BT Task: FindCoverLocation**
   - Public/BehaviorTree/BTTask_FindCoverLocation.h
   - Private/BehaviorTree/BTTask_FindCoverLocation.cpp
   - Uses EQS (Environment Query System) or raycasts to find cover
   - Writes result to Blackboard.CoverLocation
   - Returns Success if cover found, Failure otherwise

3. **Create Custom BT Task: EvasiveMovement**
   - Public/BehaviorTree/BTTask_EvasiveMovement.h
   - Private/BehaviorTree/BTTask_EvasiveMovement.cpp
   - Executes zigzag movement pattern
   - Uses AIMoveTo with randomized offsets
   - Duration: configurable (default 2-3 seconds)
   - Returns Success when complete

4. **Create Behavior Tree Flee Subtree** (in Unreal Editor after C++ compiles)
   Structure:
   ```
   FleeBehavior (Sequence)
   ├─ [Decorator: CheckStrategy == "Flee"]
   ├─ [Service: UpdateObservation]
   ├─ Task: FindCoverLocation (EQS or custom)
   ├─ Selector
   │  ├─ Sequence (if cover found)
   │  │  ├─ Task: MoveTo CoverLocation (built-in)
   │  │  └─ Task: Wait (crouch in cover, 1 sec)
   │  └─ Task: EvasiveMovement (fallback if no cover)
   └─ Task: Wait (0.5 sec before re-evaluating)
   ```

5. **Reward Function for Flee MCTS**
   Update reward calculation in MCTS for flee actions:
   ```cpp
   // In flee state
   if (CurrentState == FleeState)
   {
       float CoverReward = bHasCover ? 100.0f : 0.0f;
       float DistanceFromEnemiesReward = /* average distance to enemies */ / 50.0f;
       float HealthPreservationReward = (Health > PreviousHealth) ? 50.0f : 0.0f;
       float StaminaCostPenalty = (Stamina < 20.0f) ? -30.0f : 0.0f;

       return CoverReward + DistanceFromEnemiesReward +
              HealthPreservationReward + StaminaCostPenalty;
   }
   ```

Files to create:
- Public/BehaviorTree/BTTask_FindCoverLocation.h
- Private/BehaviorTree/BTTask_FindCoverLocation.cpp
- Public/BehaviorTree/BTTask_EvasiveMovement.h
- Private/BehaviorTree/BTTask_EvasiveMovement.cpp

Files to modify:
- Public/States/FleeState.h (add MCTS integration)
- Private/States/FleeState.cpp (implement strategic logic)
- Private/AI/MCTS.cpp (add flee-specific reward calculation)

Please provide complete implementations with proper error handling and Blueprint exposure.
```

---

### Step 2.4: Add Configurable MCTS Parameters

**Prompt:**
```
Currently, MCTS parameters are hardcoded in Private/AI/MCTS.cpp. I need to expose these as UPROPERTY for runtime tuning, allowing designers to experiment with different configurations for each strategic state.

Parameters to expose:
- ExplorationParameter (currently 1.41)
- DiscountFactor (currently 0.95)
- MaxTreeDepth (currently 10)
- SimulationCount (number of MCTS iterations per decision)
- bEnableDynamicExploration (toggle adaptive exploration)

Requirements:

1. **Update MCTS.h with UPROPERTY declarations**
   Add to Public/AI/MCTS.h:
   ```cpp
   UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MCTS|Parameters", meta = (ClampMin = "0.1", ClampMax = "5.0"))
   float ExplorationParameter = 1.41f;

   UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MCTS|Parameters", meta = (ClampMin = "0.0", ClampMax = "1.0"))
   float DiscountFactor = 0.95f;

   UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MCTS|Limits", meta = (ClampMin = "1", ClampMax = "50"))
   int32 MaxTreeDepth = 10;

   UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MCTS|Performance", meta = (ClampMin = "10", ClampMax = "10000"))
   int32 SimulationCount = 100;

   UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MCTS|Advanced")
   bool bEnableDynamicExploration = true;
   ```

2. **Add tooltips explaining each parameter**
   Use meta = (ToolTip = "...") for each UPROPERTY:
   - ExplorationParameter: "Controls exploration vs exploitation balance. Higher = more exploration (try new actions). √2 (1.41) is standard."
   - DiscountFactor: "How much to value future rewards. 0.0 = only immediate rewards, 1.0 = future rewards equally valued"
   - MaxTreeDepth: "Maximum depth of MCTS search tree. Higher = better decisions but slower performance"
   - SimulationCount: "Number of MCTS iterations per decision. Higher = better quality but more CPU time"
   - bEnableDynamicExploration: "Automatically adjust exploration based on tree depth and performance"

3. **Replace hardcoded values in MCTS.cpp**
   Update Private/AI/MCTS.cpp to use member variables instead of constants

4. **Create per-state parameter profiles (optional enhancement)**
   Allow different MCTS configs for different states:
   - FleeState might need higher exploration (try many escape routes)
   - AttackState might need lower exploration (stick to proven tactics)
   - MoveToState might need deeper trees (long-term planning)

   Add to StateMachine:
   ```cpp
   UPROPERTY(EditAnywhere, Category = "MCTS Profiles")
   TMap<TSubclassOf<UState>, UMCTSParameterProfile*> StateParameterProfiles;
   ```

5. **Add runtime parameter adjustment methods**
   ```cpp
   UFUNCTION(BlueprintCallable, Category = "MCTS")
   void SetExplorationParameter(float NewValue);

   UFUNCTION(BlueprintCallable, Category = "MCTS")
   void ResetToDefaultParameters();
   ```

Files to modify:
- Public/AI/MCTS.h
- Private/AI/MCTS.cpp

Optional files to create (for per-state profiles):
- Public/AI/MCTSParameterProfile.h
- Private/AI/MCTSParameterProfile.cpp

Please implement these changes following Unreal's UPROPERTY best practices with proper metadata for the editor.
```

---

### Step 2.5: Add Unit Tests for Core Components

**Prompt:**
```
I need to add unit tests for the SBDAPM core components using Unreal's Automation Framework to ensure reliability and catch regressions.

Create tests for:

1. **MCTS Algorithm Tests**
   - File: Private/Tests/MCTSTest.cpp
   - Tests:
     * Test_UCT_SelectionLogic: Verify UCT formula selects correct child node
     * Test_TreeExpansion: Ensure child nodes created correctly
     * Test_Backpropagation: Verify rewards propagate up tree correctly
     * Test_ObservationSimilarity: Test similarity calculation between observations
     * Test_DynamicExploration: Verify adaptive exploration parameter
     * Test_MaxDepthEnforcement: Ensure tree doesn't exceed MaxTreeDepth

2. **StateMachine Tests**
   - File: Private/Tests/StateMachineTest.cpp
   - Tests:
     * Test_StateTransitions: Verify transitions between states work correctly
     * Test_ObservationUpdates: Ensure observation updates sync correctly
     * Test_BlackboardCommunication: Verify Blackboard writes work
     * Test_GetCurrentState: Test state query functions

3. **ObservationElement Tests**
   - File: Private/Tests/ObservationElementTest.cpp
   - Tests:
     * Test_FeatureVectorSize: Ensure ToFeatureVector() returns exactly 71 values
     * Test_FeatureNormalization: Verify all values in [0, 1] range
     * Test_Reset: Ensure Reset() clears all data correctly
     * Test_EnemyObservationArray: Verify NearbyEnemies always has 5 elements

4. **Behavior Tree Integration Tests**
   - File: Private/Tests/BehaviorTreeIntegrationTest.cpp
   - Tests:
     * Test_DecoratorCheckStrategy: Verify BTDecorator_CheckStrategy reads Blackboard correctly
     * Test_ServiceUpdateObservation: Ensure BTService_UpdateObservation gathers data
     * Test_BlackboardSync: Verify Blackboard keys update when states change

Test file structure:
```cpp
#include "Misc/AutomationTest.h"
#include "AI/MCTS.h"
#include "Core/StateMachine.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMCTSUCTSelectionTest,
    "SBDAPM.MCTS.UCTSelection",
    EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter
)

bool FMCTSUCTSelectionTest::RunTest(const FString& Parameters)
{
    // Create MCTS instance
    UMCTS* MCTS = NewObject<UMCTS>();

    // Test logic here
    TestTrue("MCTS created", MCTS != nullptr);

    // Clean up
    MCTS->ConditionalBeginDestroy();

    return true;
}
```

Test organization:
- Create Private/Tests/ directory
- MCTSTest.cpp
- StateMachineTest.cpp
- ObservationElementTest.cpp
- BehaviorTreeIntegrationTest.cpp

Target at least 80% code coverage for:
- MCTS core algorithm (selection, expansion, backpropagation)
- StateMachine state transitions
- Observation data handling

Please provide complete test implementations using IMPLEMENT_SIMPLE_AUTOMATION_TEST and related macros. Include both positive tests (expected behavior) and negative tests (edge cases, error handling).
```

---

## Phase 2: Neural Network Integration (Weeks 3-4)

### Step 3.1: Design Neural Network Architecture

**Prompt:**
```
I need to implement a neural network policy for SBDAPM that works alongside MCTS (AlphaZero-style) to provide strategic guidance to the Behavior Tree system.

Architecture (from FINAL_METHODOLOGY.md):
- Input: 71 features (from FObservationElement.ToFeatureVector())
- Hidden Layer 1: 256 units, ReLU activation
- Hidden Layer 2: 256 units, ReLU activation
- Output Heads:
  * Policy Head: Softmax over strategic decisions (Attack, Flee, MoveTo, Hold)
  * Value Head: Tanh output [-1, 1] (state value estimate)

Integration approach options:
1. **Option A: Unreal's LearningAgents Plugin** (RECOMMENDED)
   - Built-in RL framework
   - PPO algorithm support
   - Distributed training infrastructure
   - Seamless integration with existing code

2. **Option B: PyTorch C++ API (libtorch)**
   - More control over architecture
   - Requires manual training pipeline
   - Need to link libtorch libraries

3. **Option C: ONNX Runtime for Inference**
   - Train model in Python (PyTorch/TensorFlow)
   - Export to ONNX format
   - Load in C++ for fast inference
   - Separate training and inference environments

Requirements:

1. **Choose integration approach** (recommend Option C for flexibility)

2. **Create Neural Network Wrapper Class**
   - Public/AI/NeuralNetworkPolicy.h
   - Private/AI/NeuralNetworkPolicy.cpp
   - Methods:
     ```cpp
     // Load trained model from file
     bool LoadModel(const FString& ModelPath);

     // Forward pass: observation → policy + value
     void Predict(const FObservationElement& Obs,
                  TArray<float>& OutPolicyProbs,  // Strategy probabilities
                  float& OutValue);                 // State value

     // Get strategic decision from policy
     FString SelectStrategy(const FObservationElement& Obs);
     ```

3. **Implement ONNX Runtime Integration** (if using Option C)
   - Link ONNX Runtime library in GameAI_Project.Build.cs
   - Create OnnxModel wrapper class
   - Handle model loading, inference, memory management

4. **Update StateMachine for NN Integration**
   Add methods:
   ```cpp
   UPROPERTY(EditAnywhere, Category = "Neural Network")
   bool bUseNeuralNetwork = false;  // Toggle MCTS vs NN vs Hybrid

   UPROPERTY(EditAnywhere, Category = "Neural Network")
   UNeuralNetworkPolicy* NeuralPolicy;

   // Get strategic decision using NN
   FString GetNeuralNetworkDecision();
   ```

5. **Hybrid MCTS + NN Mode (for future)**
   - NN provides prior probabilities
   - MCTS refines using tree search
   - Best of both worlds (AlphaZero approach)

Files to create:
- Public/AI/NeuralNetworkPolicy.h
- Private/AI/NeuralNetworkPolicy.cpp
- Public/AI/OnnxModel.h (if using ONNX)
- Private/AI/OnnxModel.cpp

Files to modify:
- GameAI_Project.Build.cs (add library dependencies)
- Public/Core/StateMachine.h (add NN integration)
- Private/Core/StateMachine.cpp

Include instructions for:
- Installing/linking required libraries (ONNX Runtime or libtorch)
- Training a dummy model in Python for testing
- Exporting model to ONNX format

Please provide complete neural network implementation with clear comments explaining the architecture and integration points.
```

---

## Quick Reference: Updated File Creation Checklist

### Phase 1 (Foundation) - Behavior Tree Architecture
- [x] Public/Core/ObservationElement.h (71 features)
- [x] Private/Core/ObservationElement.cpp
- [x] Public/Core/StateMachine.h (Blackboard integration)
- [x] Private/Core/StateMachine.cpp
- [ ] Public/AI/SBDAPMController.h (AI Controller)
- [ ] Private/AI/SBDAPMController.cpp
- [ ] Public/BehaviorTree/BTDecorator_CheckStrategy.h
- [ ] Private/BehaviorTree/BTDecorator_CheckStrategy.cpp
- [ ] Public/BehaviorTree/BTService_UpdateObservation.h
- [ ] Private/BehaviorTree/BTService_UpdateObservation.cpp
- [ ] Public/BehaviorTree/BTTask_FindCoverLocation.h
- [ ] Private/BehaviorTree/BTTask_FindCoverLocation.cpp
- [ ] Public/BehaviorTree/BTTask_EvasiveMovement.h
- [ ] Private/BehaviorTree/BTTask_EvasiveMovement.cpp
- [ ] Private/States/FleeState.cpp (complete implementation)
- [ ] Private/Tests/MCTSTest.cpp
- [ ] Private/Tests/StateMachineTest.cpp
- [ ] Private/Tests/ObservationElementTest.cpp

### Blueprint Assets (Create in Unreal Editor)
- [ ] Content/AI/BB_SBDAPM.uasset (Blackboard)
- [ ] Content/AI/BT_SBDAPM.uasset (Behavior Tree)
- [ ] Content/AI/BT_FleeBehavior.uasset (Flee subtree)
- [ ] Content/AI/BT_AttackBehavior.uasset (Attack subtree)
- [ ] Content/AI/BT_MoveToBehavior.uasset (MoveTo subtree)

### Phase 2 (Neural Networks)
- [ ] Public/AI/NeuralNetworkPolicy.h
- [ ] Private/AI/NeuralNetworkPolicy.cpp
- [ ] Public/AI/OnnxModel.h (if using ONNX)
- [ ] Private/AI/OnnxModel.cpp

---

## Estimated Timeline (Updated)

- **Phase 1 Foundation:** 2-3 weeks
  - Step 2.1: Observation Space ✅ (Completed)
  - Step 2.2: BT Integration Layer (3-4 days)
  - Step 2.3: FleeState + Flee Subtree (2-3 days)
  - Step 2.4: Configurable MCTS (1 day)
  - Step 2.5: Unit Tests (2-3 days)

- **Phase 2 Neural Networks:** 2 weeks
  - Step 3.1: NN Architecture (3-4 days)
  - Step 3.2: Hybrid MCTS + NN (4-5 days)
  - Step 3.3: Experience Replay (2-3 days)

**Key Architectural Benefits:**
✅ **Separation of Concerns:** Strategy (FSM/MCTS) vs Tactics (Behavior Tree)
✅ **Leverage Unreal Tools:** EQS, NavMesh, built-in BT tasks
✅ **Designer-Friendly:** Behavior Trees editable in Blueprint
✅ **Scalable:** Easy to add new strategies and tactical behaviors
✅ **Testable:** Unit test strategic layer, playtest tactical layer independently

---

**Generated:** 2025-10-27 (Updated for BT Architecture)
**Based on:** FINAL_METHODOLOGY.md + Behavior Tree Integration
**Status:** Ready for Implementation (Step 2.2+)
