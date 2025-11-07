// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTService.h"
#include "Observation/ObservationElement.h"
#include "BTService_UpdateObservation.generated.h"

class AAIController;

/**
 * UBTService_UpdateObservation - Behavior Tree Service for Observation Updates
 *
 * This service runs continuously while its parent node is active, gathering
 * environmental data and updating both the StateMachine's observation and
 * relevant Blackboard keys for tactical decision-making.
 *
 * Responsibilities:
 * 1. Gather agent state (position, velocity, health, etc.)
 * 2. Perform 16-ray raycasts for 360° environment perception
 * 3. Scan for nearby enemies (top 5 closest)
 * 4. Detect cover availability and positions
 * 5. Update weapon/combat state
 * 6. Call StateMachine->UpdateObservation() with new data
 * 7. Sync key values to Blackboard for BT tasks to use
 *
 * Update Frequency:
 * - Configurable via UpdateInterval property (default: 0.1 seconds / 10 Hz)
 * - Critical for maintaining up-to-date tactical awareness
 * - Balance between responsiveness and performance
 *
 * Integration:
 * - Attach this service to behavior tree nodes that need continuous perception
 * - Typically placed on subtree root nodes (Attack, Flee, MoveTo behaviors)
 * - Works in conjunction with BTDecorator_CheckStrategy for strategy-based execution
 */
UCLASS()
class GAMEAI_PROJECT_API UBTService_UpdateObservation : public UBTService
{
	GENERATED_BODY()

public:
	UBTService_UpdateObservation();

protected:
	/**
	 * Called when the service becomes active (parent node starts executing).
	 */
	virtual void OnBecomeRelevant(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;

	/**
	 * Called periodically while the service is active.
	 * This is where we gather observation data and update the StateMachine.
	 */
	virtual void TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;

	/**
	 * Called when the service becomes inactive (parent node stops executing).
	 */
	virtual void OnCeaseRelevant(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;

public:
	/**
	 * How often to update the observation (in seconds).
	 * Default: 0.1 seconds (10 updates per second)
	 *
	 * Lower values = more responsive but higher CPU cost
	 * Higher values = less responsive but lower CPU cost
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Observation", meta = (ClampMin = "0.01", ClampMax = "1.0"))
	float UpdateInterval = 0.1f;

	/**
	 * Maximum distance for enemy detection (in Unreal units, typically cm).
	 * Default: 3000.0 (30 meters)
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Observation|Perception", meta = (ClampMin = "100.0", ClampMax = "10000.0"))
	float MaxEnemyDetectionDistance = 3000.0f;

	/**
	 * Tag used to identify enemy actors.
	 * Enemies should be tagged with this in the editor.
	 * Default: "Enemy"
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Observation|Perception")
	FName EnemyTag = FName("Enemy");

	/**
	 * Maximum distance for raycast environment perception (in Unreal units).
	 * Default: 2000.0 (20 meters)
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Observation|Perception", meta = (ClampMin = "100.0", ClampMax = "5000.0"))
	float RaycastMaxDistance = 2000.0f;

	/**
	 * Number of raycasts to perform around the agent.
	 * Default: 16 (360° / 16 = 22.5° between each ray)
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Observation|Perception", meta = (ClampMin = "4", ClampMax = "32"))
	int32 RaycastCount = 16;

	/**
	 * Maximum distance for cover detection (in Unreal units).
	 * Default: 1500.0 (15 meters)
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Observation|Tactical", meta = (ClampMin = "100.0", ClampMax = "5000.0"))
	float CoverDetectionDistance = 1500.0f;

	/**
	 * Tag used to identify cover actors/objects.
	 * Cover objects should be tagged with this in the editor.
	 * Default: "Cover"
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Observation|Tactical")
	FName CoverTag = FName("Cover");

	/**
	 * Whether to draw debug visualizations for raycasts and detections.
	 * Useful for debugging perception issues.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
	bool bDrawDebugInfo = false;

	/**
	 * Whether to enable verbose logging for observation updates.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
	bool bEnableDebugLog = false;

private:
	/**
	 * Gather all observation data for the agent.
	 * Returns a fully populated FObservationElement struct.
	 */
	FObservationElement GatherObservationData(AAIController* OwnerController, APawn* ControlledPawn);

	/**
	 * Update agent state information (position, velocity, health, etc.).
	 */
	void UpdateAgentState(FObservationElement& Observation, APawn* ControlledPawn);

	/**
	 * Perform raycast-based environment perception.
	 * Fills in RaycastDistances and RaycastHitTypes arrays.
	 */
	void PerformRaycastPerception(FObservationElement& Observation, APawn* ControlledPawn);

	/**
	 * Scan for nearby enemies and populate NearbyEnemies array.
	 */
	void ScanForEnemies(FObservationElement& Observation, APawn* ControlledPawn);

	/**
	 * Detect cover availability and nearest cover position.
	 */
	void DetectCover(FObservationElement& Observation, APawn* ControlledPawn);

	/**
	 * Update combat-related state (weapon, ammo, cooldowns).
	 */
	void UpdateCombatState(FObservationElement& Observation, APawn* ControlledPawn);

	/**
	 * Sync observation data to Blackboard keys for BT tasks to use.
	 */
	void SyncToBlackboard(UBlackboardComponent* BlackboardComp, const FObservationElement& Observation);

	/**
	 * Classify what type of object a raycast hit.
	 */
	ERaycastHitType ClassifyHitType(const FHitResult& HitResult);

	/**
	 * Detect terrain type based on surface normal and slope angle.
	 */
	ETerrainType DetectTerrainType(APawn* ControlledPawn);

	/**
	 * Time accumulator for update interval timing.
	 */
	float TimeAccumulator = 0.0f;
};
