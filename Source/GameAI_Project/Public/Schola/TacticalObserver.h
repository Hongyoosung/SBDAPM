// TacticalObserver.h - Schola observer for 71-feature tactical observation

#pragma once

#include "CoreMinimal.h"
#include "Observers/AbstractObservers.h"
#include "TacticalObserver.generated.h"

class UFollowerAgentComponent;

/**
 * Schola observer that exposes 78-feature tactical observation for RL training.
 * Used for live training via gRPC connection to Python/RLlib.
 *
 * Total: 78 features = 71 tactical + 7 objective embedding
 *
 * Tactical Features (71 total):
 * - Agent State: Position, Velocity, Rotation, Health, Stamina, Shield (12 features)
 * - Combat State: WeaponCooldown, Ammunition, WeaponType (3 features)
 * - Environment: RaycastDistances, RaycastHitTypes (32 features)
 * - Enemies: VisibleEnemyCount, NearbyEnemies (16 features)
 * - Tactical: Cover info, Terrain (5 features)
 * - Temporal: TimeSinceLastAction, LastActionType (2 features)
 * - Combat Proximity: DistanceToNearestEnemy (1 feature)
 *
 * State Embedding (7 features, one-hot):
 * - [71]: Idle, [72]: Assault, [73]: Defend, [74]: Support, [75]: Move, [76]: Retreat, [77]: Dead
 */
UCLASS(BlueprintType, meta = (DisplayName = "Tactical Observer"))
class GAMEAI_PROJECT_API UTacticalObserver : public UBoxObserver
{
	GENERATED_BODY()

public:
	UTacticalObserver();

	// UBoxObserver interface
	virtual FBoxSpace GetObservationSpace() const override;
	virtual void CollectObservations(FBoxPoint& OutObservations) override;
	virtual void InitializeObserver() override;
	virtual void ResetObserver() override;

	/** The follower agent component to observe */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Observer")
	UFollowerAgentComponent* FollowerAgent = nullptr;

	/** Auto-find follower agent on owner actor */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Observer")
	bool bAutoFindFollower = true;

protected:
	/** Cached observation space */
	FBoxSpace CachedObservationSpace;

	/** Find follower agent component */
	UFollowerAgentComponent* FindFollowerAgent() const;
};
