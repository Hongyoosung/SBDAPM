#pragma once

#include "CoreMinimal.h"
#include "EnvironmentQuery/EnvQueryTest.h"
#include "EnvQueryTest_CoverQuality.generated.h"

/**
 * EQS Test that evaluates cover quality based on multiple factors:
 * - Distance from enemies (prefer farther)
 * - Line of sight to enemies (prefer blocked)
 * - Distance from querier (prefer closer)
 * - Navigability (must be reachable)
 */
UCLASS()
class GAMEAI_PROJECT_API UEnvQueryTest_CoverQuality : public UEnvQueryTest
{
	GENERATED_BODY()

public:
	UEnvQueryTest_CoverQuality();

	virtual void RunTest(FEnvQueryInstance& QueryInstance) const override;

	virtual FText GetDescriptionTitle() const override;
	virtual FText GetDescriptionDetails() const override;


private:
	/** Calculate cover score for a location */
	float CalculateCoverScore(
		const FVector& CoverLocation,
		const FVector& QuerierLocation,
		const TArray<FVector>& EnemyLocations,
		UWorld* World
	) const;

	/** Check if there's line of sight between two points */
	bool HasLineOfSight(const FVector& From, const FVector& To, UWorld* World) const;


public:
	/** Weight for enemy distance scoring (0-1) */
	UPROPERTY(EditDefaultsOnly, Category = "Cover Quality", meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float EnemyDistanceWeight = 0.5f;

	/** Weight for line of sight scoring (0-1) */
	UPROPERTY(EditDefaultsOnly, Category = "Cover Quality", meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float LineOfSightWeight = 0.3f;

	/** Weight for distance from querier (0-1) */
	UPROPERTY(EditDefaultsOnly, Category = "Cover Quality", meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float QuerierDistanceWeight = 0.2f;

	/** Minimum safe distance from enemies */
	UPROPERTY(EditDefaultsOnly, Category = "Cover Quality")
	float MinSafeDistance = 500.0f;

	/** Maximum useful distance from enemies (beyond this, distance doesn't matter) */
	UPROPERTY(EditDefaultsOnly, Category = "Cover Quality")
	float MaxSafeDistance = 2000.0f;

	/** Whether to check navigability */
	UPROPERTY(EditDefaultsOnly, Category = "Cover Quality")
	bool bCheckNavigability = true;

	/** Whether to check line of sight to enemies */
	UPROPERTY(EditDefaultsOnly, Category = "Cover Quality")
	bool bCheckLineOfSight = true;

	/** Context providing enemy locations for cover evaluation */
	UPROPERTY(EditDefaultsOnly, Category = "Cover Quality")
	TSubclassOf<UEnvQueryContext> EnemyContext;
};
