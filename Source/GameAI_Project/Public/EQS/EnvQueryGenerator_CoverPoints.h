#pragma once

#include "CoreMinimal.h"
#include "EnvironmentQuery/EnvQueryGenerator.h"
#include "EnvQueryGenerator_CoverPoints.generated.h"

/**
 * Generates potential cover points around the querier
 * Uses grid-based sampling with optional actor-based cover locations
 */
UCLASS()
class GAMEAI_PROJECT_API UEnvQueryGenerator_CoverPoints : public UEnvQueryGenerator
{
	GENERATED_BODY()

public:
	UEnvQueryGenerator_CoverPoints();

	virtual void GenerateItems(FEnvQueryInstance& QueryInstance) const override;

	virtual FText GetDescriptionTitle() const override;
	virtual FText GetDescriptionDetails() const override;

	/** Search radius for cover points */
	UPROPERTY(EditDefaultsOnly, Category = "Generator", meta = (ClampMin = "100.0", ClampMax = "5000.0"))
	float SearchRadius = 1500.0f;

	/** Grid spacing for sampling points */
	UPROPERTY(EditDefaultsOnly, Category = "Generator", meta = (ClampMin = "50.0", ClampMax = "500.0"))
	float GridSpacing = 200.0f;

	/** Tag to find pre-placed cover actors */
	UPROPERTY(EditDefaultsOnly, Category = "Generator")
	FName CoverTag = FName("Cover");

	/** Whether to include tagged cover actors */
	UPROPERTY(EditDefaultsOnly, Category = "Generator")
	bool bIncludeTaggedCover = true;

	/** Whether to generate grid points */
	UPROPERTY(EditDefaultsOnly, Category = "Generator")
	bool bGenerateGridPoints = true;

	/** Trace down distance to project points to ground */
	UPROPERTY(EditDefaultsOnly, Category = "Generator")
	float GroundTraceDistance = 500.0f;
};
