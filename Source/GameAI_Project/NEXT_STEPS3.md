
  # Step 2.3: Implement Flee Behavior Tree Subtree & FleeState Logic

  Now that we have the Behavior Tree integration layer working (Step 2.2 completed), I need to
  complete the FleeState strategic logic and create the corresponding tactical Behavior Tree
  subtree with custom tasks.

  ## Context

  **What's Already Done:**
  - ✅ SBDAPMController (AI Controller with BT management)
  - ✅ BTDecorator_CheckStrategy (strategy-based subtree activation)
  - ✅ BTService_UpdateObservation (observation gathering)
  - ✅ StateMachine Blackboard helper methods
  - ✅ Basic FleeState with Blackboard writes (sets strategy, updates cover location)

  **What's Missing:**
  - FleeState needs full MCTS integration for strategic decision-making
  - No custom Behavior Tree tasks for flee execution (FindCoverLocation, EvasiveMovement)
  - No flee-specific actions for MCTS to evaluate
  - No reward function for flee behavior evaluation

  ## Requirements

  ### 1. Complete FleeState Strategic Logic

  **File:** `Private/States/FleeState.cpp` and `Public/States/FleeState.h`

  FleeState should use MCTS to decide:
  - Which flee strategy? (Sprint to cover, Evasive movement, Fight while retreating)
  - Which cover location? (Use NearestCoverDistance and CoverDirection from observation)
  - When to stop fleeing? (Exit conditions)

  **Implementation Details:**

  **In FleeState.h:**
  - Add MCTS instance member variable (like MoveToState and AttackState have)
  - Add PossibleActions array

  **In FleeState.cpp:**

  **EnterState():**
  ```cpp
  void UFleeState::EnterState(UStateMachine* StateMachine)
  {
      Super::EnterState(StateMachine);
      UE_LOG(LogTemp, Warning, TEXT("Entered FleeState"));

      // Set Blackboard strategy (already done)
      StateMachine->SetCurrentStrategy(TEXT("Flee"));

      // Initialize MCTS for flee decision-making
      if (MCTS == nullptr)
      {
          MCTS = NewObject<UMCTS>();
          MCTS->InitializeMCTS();
          MCTS->InitializeCurrentNodeLocate();
          PossibleActions = GetPossibleActions();
      }
      else
      {
          MCTS->InitializeCurrentNodeLocate();
      }
  }

  UpdateState():
  void UFleeState::UpdateState(UStateMachine* StateMachine, float Reward, float DeltaTime)
  {
      if (!MCTS)
      {
          UE_LOG(LogTemp, Error, TEXT("FleeState: MCTS is nullptr"));
          return;
      }

      // Run MCTS to select optimal flee strategy
      MCTS->RunMCTS(PossibleActions, StateMachine);

      // Update Blackboard based on MCTS decision and observations
      if (StateMachine)
      {
          FObservationElement CurrentObs = StateMachine->GetCurrentObservation();

          if (CurrentObs.bHasCover)
          {
              // Calculate and set cover location (already implemented)
              // ... existing cover location code ...
          }

          // Update threat level
          float ThreatLevel = FMath::Clamp(CurrentObs.VisibleEnemyCount / 10.0f, 0.0f, 1.0f);
          StateMachine->SetThreatLevel(ThreatLevel);
      }
  }

  ExitState():
  void UFleeState::ExitState(UStateMachine* StateMachine)
  {
      if (MCTS)
      {
          MCTS->Backpropagate();
          UE_LOG(LogTemp, Warning, TEXT("Exited FleeState"));
      }
  }

  GetPossibleActions():
  Create flee-specific actions that MCTS can evaluate:
  TArray<UAction*> UFleeState::GetPossibleActions()
  {
      TArray<UAction*> Actions;

      // Create flee-specific actions
      // These represent strategic choices, not actual movement
      // The Behavior Tree will handle the tactical execution

      // Note: You'll need to create these action classes
      // They can be simple classes that just represent the strategy choice
      Actions.Add(NewObject<USprintToCoverAction>(this, USprintToCoverAction::StaticClass()));
      Actions.Add(NewObject<UEvasiveMovementAction>(this, UEvasiveMovementAction::StaticClass()));     
      Actions.Add(NewObject<UFightWhileRetreatingAction>(this,
  UFightWhileRetreatingAction::StaticClass()));

      return Actions;
  }

  State Transition Logic:

  The FSM should transition to FleeState when:
  - Health < 20% AND VisibleEnemyCount > 3, OR
  - Health < 30% AND VisibleEnemyCount > 5, OR
  - Health < 40% AND bHasCover == false AND VisibleEnemyCount > 2

  Exit FleeState when:
  - No enemies within 2000 units (safe distance reached), OR
  - Health restored > 40%, OR
  - Reached cover AND Health > 25%

  Note: This transition logic should be implemented wherever you handle state transitions (possibly    
   in a separate state manager or in UpdateState checks).

  ---
  2. Create Flee Action Classes (Simple Strategic Representations)

  These are simple action classes that represent strategic choices for MCTS. They don't need to do     
  actual movement - that's handled by the Behavior Tree.

  Files to create:
  - Public/Actions/FleeActions/SprintToCoverAction.h
  - Private/Actions/FleeActions/SprintToCoverAction.cpp
  - Public/Actions/FleeActions/EvasiveMovementAction.h
  - Private/Actions/FleeActions/EvasiveMovementAction.cpp
  - Public/Actions/FleeActions/FightWhileRetreatingAction.h
  - Private/Actions/FleeActions/FightWhileRetreatingAction.cpp

  Example (SprintToCoverAction.h):
  #pragma once

  #include "CoreMinimal.h"
  #include "Actions/Action.h"
  #include "SprintToCoverAction.generated.h"

  UCLASS()
  class GAMEAI_PROJECT_API USprintToCoverAction : public UAction
  {
      GENERATED_BODY()

  public:
      USprintToCoverAction();

      virtual void Execute(UStateMachine* StateMachine) override;
      virtual FString GetActionName() const override { return TEXT("Sprint to Cover"); }
  };

  Example (SprintToCoverAction.cpp):
  #include "Actions/FleeActions/SprintToCoverAction.h"
  #include "Core/StateMachine.h"

  USprintToCoverAction::USprintToCoverAction()
  {
      ActionName = TEXT("Sprint to Cover");
  }

  void USprintToCoverAction::Execute(UStateMachine* StateMachine)
  {
      // This is a strategic action - actual execution handled by Behavior Tree
      // Just log or trigger Blueprint event if needed
      UE_LOG(LogTemp, Log, TEXT("FleeAction: Sprint to Cover selected by MCTS"));

      // The Behavior Tree's Flee subtree will handle the actual sprinting
      // using the CoverLocation already set on the Blackboard
  }

  Create similar implementations for EvasiveMovementAction and FightWhileRetreatingAction.

  ---
  3. Create Custom BT Task: FindCoverLocation

  Files:
  - Public/BehaviorTree/BTTask_FindCoverLocation.h
  - Private/BehaviorTree/BTTask_FindCoverLocation.cpp

  Purpose: Uses raycasts or EQS to find cover and writes result to Blackboard.

  Implementation:

  BTTask_FindCoverLocation.h:
  #pragma once

  #include "CoreMinimal.h"
  #include "BehaviorTree/BTTaskNode.h"
  #include "BTTask_FindCoverLocation.generated.h"

  /**
   * Finds a suitable cover location and writes it to the Blackboard.
   *
   * This task searches for cover within a specified radius and evaluates
   * positions based on:
   * - Distance from enemies (further = safer)
   * - Availability of line-of-sight blocking
   * - Distance from agent (closer = faster to reach)
   */
  UCLASS()
  class GAMEAI_PROJECT_API UBTTask_FindCoverLocation : public UBTTaskNode
  {
      GENERATED_BODY()

  public:
      UBTTask_FindCoverLocation();

  protected:
      virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)    
   override;
      virtual FString GetStaticDescription() const override;

  public:
      /** Maximum search radius for cover (in cm) */
      UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cover", meta = (ClampMin = "100.0",      
  ClampMax = "5000.0"))
      float SearchRadius = 1500.0f;

      /** Blackboard key to write the cover location to */
      UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blackboard")
      FName CoverLocationKey = FName("CoverLocation");

      /** Tag used to identify cover objects */
      UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cover")
      FName CoverTag = FName("Cover");

      /** Whether to draw debug visualization */
      UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
      bool bDrawDebug = false;

  private:
      /** Find the best cover location */
      bool FindBestCover(APawn* ControlledPawn, FVector& OutCoverLocation);
  };

  BTTask_FindCoverLocation.cpp:
  #include "BehaviorTree/BTTask_FindCoverLocation.h"
  #include "BehaviorTree/BlackboardComponent.h"
  #include "AIController.h"
  #include "Kismet/GameplayStatics.h"
  #include "DrawDebugHelpers.h"

  UBTTask_FindCoverLocation::UBTTask_FindCoverLocation()
  {
      NodeName = "Find Cover Location";
      bNotifyTick = false;
  }

  EBTNodeResult::Type UBTTask_FindCoverLocation::ExecuteTask(UBehaviorTreeComponent& OwnerComp,        
  uint8* NodeMemory)
  {
      AAIController* AIController = OwnerComp.GetAIOwner();
      if (!AIController)
      {
          return EBTNodeResult::Failed;
      }

      APawn* ControlledPawn = AIController->GetPawn();
      if (!ControlledPawn)
      {
          return EBTNodeResult::Failed;
      }

      FVector CoverLocation;
      if (FindBestCover(ControlledPawn, CoverLocation))
      {
          // Write cover location to Blackboard
          UBlackboardComponent* BlackboardComp = OwnerComp.GetBlackboardComponent();
          if (BlackboardComp)
          {
              BlackboardComp->SetValueAsVector(CoverLocationKey, CoverLocation);
              UE_LOG(LogTemp, Log, TEXT("BTTask_FindCoverLocation: Found cover at %s"),
  *CoverLocation.ToString());
              return EBTNodeResult::Succeeded;
          }
      }

      UE_LOG(LogTemp, Warning, TEXT("BTTask_FindCoverLocation: No cover found within radius %.1f"),    
   SearchRadius);
      return EBTNodeResult::Failed;
  }

  bool UBTTask_FindCoverLocation::FindBestCover(APawn* ControlledPawn, FVector& OutCoverLocation)      
  {
      UWorld* World = ControlledPawn->GetWorld();
      if (!World)
      {
          return false;
      }

      // Find all cover actors within search radius
      TArray<AActor*> CoverActors;
      UGameplayStatics::GetAllActorsWithTag(World, CoverTag, CoverActors);

      FVector AgentLocation = ControlledPawn->GetActorLocation();
      float BestScore = -1.0f;
      FVector BestCoverLocation = FVector::ZeroVector;
      bool bFoundCover = false;

      for (AActor* CoverActor : CoverActors)
      {
          if (!CoverActor)
          {
              continue;
          }

          FVector CoverLocation = CoverActor->GetActorLocation();
          float Distance = FVector::Dist(AgentLocation, CoverLocation);

          // Only consider cover within search radius
          if (Distance <= SearchRadius)
          {
              // Simple scoring: prefer closer cover
              // TODO: Factor in enemy positions for better scoring
              float Score = SearchRadius - Distance;

              if (Score > BestScore)
              {
                  BestScore = Score;
                  BestCoverLocation = CoverLocation;
                  bFoundCover = true;

                  if (bDrawDebug)
                  {
                      DrawDebugSphere(World, CoverLocation, 100.0f, 12, FColor::Green, false,
  2.0f);
                  }
              }
          }
      }

      if (bFoundCover)
      {
          OutCoverLocation = BestCoverLocation;
          return true;
      }

      return false;
  }

  FString UBTTask_FindCoverLocation::GetStaticDescription() const
  {
      return FString::Printf(TEXT("Find cover within %.1f units"), SearchRadius);
  }

  ---
  4. Create Custom BT Task: EvasiveMovement

  Files:
  - Public/BehaviorTree/BTTask_EvasiveMovement.h
  - Private/BehaviorTree/BTTask_EvasiveMovement.cpp

  Purpose: Executes zigzag movement pattern when no cover is available.

  Implementation:

  BTTask_EvasiveMovement.h:
  #pragma once

  #include "CoreMinimal.h"
  #include "BehaviorTree/BTTaskNode.h"
  #include "BTTask_EvasiveMovement.generated.h"

  /**
   * Performs evasive zigzag movement to avoid enemy fire.
   *
   * This task makes the agent move in a zigzag pattern by randomly
   * offsetting movement directions. Useful when no cover is available.
   */
  UCLASS()
  class GAMEAI_PROJECT_API UBTTask_EvasiveMovement : public UBTTaskNode
  {
      GENERATED_BODY()

  public:
      UBTTask_EvasiveMovement();

  protected:
      virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)    
   override;
      virtual void TickTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float
  DeltaSeconds) override;
      virtual FString GetStaticDescription() const override;

  public:
      /** Duration of evasive movement (in seconds) */
      UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Movement", meta = (ClampMin = "0.5",     
  ClampMax = "10.0"))
      float Duration = 2.0f;

      /** Distance for each evasive move */
      UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Movement", meta = (ClampMin =
  "100.0", ClampMax = "1000.0"))
      float MoveDistance = 300.0f;

      /** Whether to draw debug visualization */
      UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
      bool bDrawDebug = false;

  private:
      float ElapsedTime;
      float LastMoveTime;
      float MoveInterval = 0.5f; // How often to change direction
  };

  BTTask_EvasiveMovement.cpp:
  #include "BehaviorTree/BTTask_EvasiveMovement.h"
  #include "AIController.h"
  #include "NavigationSystem.h"
  #include "DrawDebugHelpers.h"

  UBTTask_EvasiveMovement::UBTTask_EvasiveMovement()
  {
      NodeName = "Evasive Movement";
      bNotifyTick = true; // We need tick to track duration
  }

  EBTNodeResult::Type UBTTask_EvasiveMovement::ExecuteTask(UBehaviorTreeComponent& OwnerComp,
  uint8* NodeMemory)
  {
      AAIController* AIController = OwnerComp.GetAIOwner();
      if (!AIController)
      {
          return EBTNodeResult::Failed;
      }

      APawn* ControlledPawn = AIController->GetPawn();
      if (!ControlledPawn)
      {
          return EBTNodeResult::Failed;
      }

      // Reset timers
      ElapsedTime = 0.0f;
      LastMoveTime = 0.0f;

      UE_LOG(LogTemp, Log, TEXT("BTTask_EvasiveMovement: Starting evasive movement for %.1f
  seconds"), Duration);

      // Task will continue in TickTask
      return EBTNodeResult::InProgress;
  }

  void UBTTask_EvasiveMovement::TickTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory,
  float DeltaSeconds)
  {
      AAIController* AIController = OwnerComp.GetAIOwner();
      if (!AIController)
      {
          FinishLatentTask(OwnerComp, EBTNodeResult::Failed);
          return;
      }

      APawn* ControlledPawn = AIController->GetPawn();
      if (!ControlledPawn)
      {
          FinishLatentTask(OwnerComp, EBTNodeResult::Failed);
          return;
      }

      ElapsedTime += DeltaSeconds;
      LastMoveTime += DeltaSeconds;

      // Check if duration has elapsed
      if (ElapsedTime >= Duration)
      {
          UE_LOG(LogTemp, Log, TEXT("BTTask_EvasiveMovement: Completed evasive movement"));
          FinishLatentTask(OwnerComp, EBTNodeResult::Succeeded);
          return;
      }

      // Change direction every MoveInterval seconds
      if (LastMoveTime >= MoveInterval)
      {
          LastMoveTime = 0.0f;

          // Calculate random evasive direction
          FVector CurrentLocation = ControlledPawn->GetActorLocation();
          FVector ForwardVector = ControlledPawn->GetActorForwardVector();
          FVector RightVector = ControlledPawn->GetActorRightVector();

          // Random zigzag: alternate left/right with some randomness
          float RandomAngle = FMath::RandRange(-90.0f, 90.0f);
          FVector RandomDirection = ForwardVector.RotateAngleAxis(RandomAngle, FVector::UpVector);     
          RandomDirection.Normalize();

          FVector TargetLocation = CurrentLocation + (RandomDirection * MoveDistance);

          // Use navigation system to move
          UNavigationSystemV1* NavSys =
  UNavigationSystemV1::GetCurrent(ControlledPawn->GetWorld());
          if (NavSys)
          {
              FNavLocation NavLocation;
              if (NavSys->ProjectPointToNavigation(TargetLocation, NavLocation))
              {
                  AIController->MoveToLocation(NavLocation.Location);

                  if (bDrawDebug)
                  {
                      DrawDebugLine(ControlledPawn->GetWorld(), CurrentLocation,
  NavLocation.Location,
                                    FColor::Yellow, false, MoveInterval, 0, 2.0f);
                      DrawDebugSphere(ControlledPawn->GetWorld(), NavLocation.Location, 50.0f, 8,      
                                     FColor::Orange, false, MoveInterval);
                  }
              }
          }
      }
  }

  FString UBTTask_EvasiveMovement::GetStaticDescription() const
  {
      return FString::Printf(TEXT("Zigzag for %.1fs"), Duration);
  }

  ---
  5. Update MCTS Reward Function for Flee Behavior

  File: Private/AI/MCTS.cpp

  Add flee-specific reward calculation logic. You'll need to check which state is active and apply     
  appropriate rewards.

  In the reward calculation method (likely in CalculateImmediateReward or similar):

  // Check if we're in FleeState (you'll need a way to determine current state)
  // This might require passing state information or checking the StateMachine
  if (CurrentStateIsFleeState) // Implement this check based on your architecture
  {
      FObservationElement Obs = CurrentObservation;

      float CoverReward = Obs.bHasCover ? 100.0f : 0.0f;

      // Calculate average distance to enemies
      float TotalEnemyDistance = 0.0f;
      int32 ValidEnemies = 0;
      for (const FEnemyObservation& Enemy : Obs.NearbyEnemies)
      {
          if (Enemy.Distance < 3000.0f) // Only count nearby enemies
          {
              TotalEnemyDistance += Enemy.Distance;
              ValidEnemies++;
          }
      }
      float AvgEnemyDistance = ValidEnemies > 0 ? TotalEnemyDistance / ValidEnemies : 3000.0f;
      float DistanceFromEnemiesReward = AvgEnemyDistance / 50.0f; // Normalize

      // Reward health preservation
      float HealthPreservationReward = (Obs.Health > PreviousHealth) ? 50.0f : 0.0f;

      // Penalize low stamina (can't sprint effectively)
      float StaminaCostPenalty = (Obs.Stamina < 20.0f) ? -30.0f : 0.0f;

      return CoverReward + DistanceFromEnemiesReward +
             HealthPreservationReward + StaminaCostPenalty;
  }

  ---
  6. Behavior Tree Structure (Blueprint - Create in Unreal Editor)

  File: Content/AI/BT_SBDAPM.uasset (modify the existing tree)

  Add the Flee subtree:

  Root (Selector)
  ├─ [Decorator: CheckStrategy == "Dead"] DeadBehavior
  │  └─ Task: Wait (play death animation)
  │
  ├─ [Decorator: CheckStrategy == "Flee"] FleeBehavior ← NEW
  │  ├─ [Service: UpdateObservation]
  │  ├─ Selector
  │  │  ├─ Sequence (Try cover first)
  │  │  │  ├─ Task: FindCoverLocation
  │  │  │  ├─ Task: MoveTo CoverLocation (built-in)
  │  │  │  └─ Task: Wait (crouch in cover, 1 sec)
  │  │  └─ Task: EvasiveMovement (fallback if no cover)
  │  └─ Task: Wait (0.5 sec before re-evaluating)
  │
  ├─ [Decorator: CheckStrategy == "Attack"] AttackBehavior
  │  └─ ... existing attack logic ...
  │
  └─ [Decorator: CheckStrategy == "MoveTo"] MoveToBehavior
     └─ ... existing movement logic ...

  ---
  Files Summary

  Files to Create (8 new files):

  Flee Actions (Strategic):
  1. Public/Actions/FleeActions/SprintToCoverAction.h
  2. Private/Actions/FleeActions/SprintToCoverAction.cpp
  3. Public/Actions/FleeActions/EvasiveMovementAction.h
  4. Private/Actions/FleeActions/EvasiveMovementAction.cpp
  5. Public/Actions/FleeActions/FightWhileRetreatingAction.h
  6. Private/Actions/FleeActions/FightWhileRetreatingAction.cpp

  BT Tasks (Tactical):
  7. Public/BehaviorTree/BTTask_FindCoverLocation.h
  8. Private/BehaviorTree/BTTask_FindCoverLocation.cpp
  9. Public/BehaviorTree/BTTask_EvasiveMovement.h
  10. Private/BehaviorTree/BTTask_EvasiveMovement.cpp

  Files to Modify:

  1. Public/States/FleeState.h - Add MCTS member, PossibleActions
  2. Private/States/FleeState.cpp - Complete MCTS integration
  3. Private/AI/MCTS.cpp - Add flee-specific reward calculation

  Blueprint Assets to Modify:

  1. Content/AI/BT_SBDAPM.uasset - Add Flee subtree with new tasks

  ---
  Expected Behavior After Implementation

  1. When agent takes damage and enemies are numerous:
    - FSM transitions to FleeState
    - FleeState MCTS evaluates flee strategies
    - Blackboard updated with strategy="Flee" and CoverLocation
    - BT Flee subtree activates
  2. If cover is available:
    - FindCoverLocation task succeeds
    - Agent uses MoveTo to reach cover
    - Agent crouches/waits at cover
  3. If no cover:
    - FindCoverLocation task fails
    - Selector falls back to EvasiveMovement
    - Agent performs zigzag movement for Duration seconds
  4. When safe:
    - FSM exits FleeState (based on transition conditions)
    - BT switches to different subtree (MoveTo or Attack)

  ---
  Testing Checklist

  After implementation:
  - FleeState enters correctly when health/enemy conditions met
  - MCTS runs in FleeState without errors
  - Cover location written to Blackboard when available
  - FindCoverLocation task finds cover actors (tagged with "Cover")
  - Agent moves to cover using MoveTo
  - EvasiveMovement activates when no cover found
  - Agent performs zigzag movement for specified duration
  - FleeState exits when conditions met (safe distance, health recovered)
  - Behavior Tree switches subtrees correctly

  ---
  Please implement all C++ components with proper error handling, logging, and Blueprint exposure.     
  Include detailed comments explaining the flee strategy logic and integration between strategic       
  (MCTS) and tactical (BT) layers.
