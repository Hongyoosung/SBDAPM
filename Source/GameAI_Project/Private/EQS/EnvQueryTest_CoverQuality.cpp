#include "EQS/EnvQueryTest_CoverQuality.h"
#include "EnvironmentQuery/Contexts/EnvQueryContext_Querier.h"
#include "EnvironmentQuery/Items/EnvQueryItemType_Point.h"
#include "EQS/EnvQueryContext_CoverEnemies.h"
#include "NavigationSystem.h"
#include "NavigationPath.h"

UEnvQueryTest_CoverQuality::UEnvQueryTest_CoverQuality()
{
	Cost = EEnvTestCost::High;
	ValidItemType = UEnvQueryItemType_Point::StaticClass();
	SetWorkOnFloatValues(false);

	EnemyDistanceWeight = 0.5f;
	LineOfSightWeight = 0.3f;
	QuerierDistanceWeight = 0.2f;
	MinSafeDistance = 500.0f;
	MaxSafeDistance = 2000.0f;
	bCheckNavigability = true;
	bCheckLineOfSight = true;

	// Default to CoverEnemies context
	EnemyContext = UEnvQueryContext_CoverEnemies::StaticClass();
}

void UEnvQueryTest_CoverQuality::RunTest(FEnvQueryInstance& QueryInstance) const
{
	UWorld* World = QueryInstance.World;
	if (!World)
	{
		return;
	}

	// Get querier location
	TArray<FVector> QuerierLocations;
	QueryInstance.PrepareContext(UEnvQueryContext_Querier::StaticClass(), QuerierLocations);
	if (QuerierLocations.Num() == 0)
	{
		return;
	}
	const FVector& QuerierLocation = QuerierLocations[0];

	// Get enemy locations from configured context
	TArray<FVector> EnemyLocations;
	if (EnemyContext)
	{
		QueryInstance.PrepareContext(EnemyContext, EnemyLocations);
	}

	// If no enemies, just use distance from querier
	if (EnemyLocations.Num() == 0)
	{
		for (FEnvQueryInstance::ItemIterator It(this, QueryInstance); It; ++It)
		{
			const FVector ItemLocation = GetItemLocation(QueryInstance, It.GetIndex());
			float Distance = FVector::Dist(QuerierLocation, ItemLocation);

			// Prefer closer positions when no enemies
			float Score = 1.0f - FMath::Clamp(Distance / MaxSafeDistance, 0.0f, 1.0f);
			It.SetScore(TestPurpose, FilterType, Score, MinSafeDistance, MaxSafeDistance);
		}
		return;
	}

	// Evaluate each potential cover location
	for (FEnvQueryInstance::ItemIterator It(this, QueryInstance); It; ++It)
	{
		const FVector ItemLocation = GetItemLocation(QueryInstance, It.GetIndex());

		// Check navigability first (hard filter)
		if (bCheckNavigability)
		{
			UNavigationSystemV1* NavSys = FNavigationSystem::GetCurrent<UNavigationSystemV1>(World);
			if (NavSys)
			{
				FPathFindingQuery Query;
				Query.StartLocation = QuerierLocation;
				Query.EndLocation = ItemLocation;
				Query.NavAgentProperties = FNavAgentProperties::DefaultProperties;

				FPathFindingResult Result = NavSys->FindPathSync(Query);
				if (!Result.IsSuccessful())
				{
					It.ForceItemState(EEnvItemStatus::Failed);
					continue;
				}
			}
		}

		// Calculate composite cover score
		float Score = CalculateCoverScore(ItemLocation, QuerierLocation, EnemyLocations, World);
		It.SetScore(TestPurpose, FilterType, Score, 0.0f, 1.0f);
	}
}

float UEnvQueryTest_CoverQuality::CalculateCoverScore(
	const FVector& CoverLocation,
	const FVector& QuerierLocation,
	const TArray<FVector>& EnemyLocations,
	UWorld* World
) const
{
	float TotalScore = 0.0f;
	float TotalWeight = 0.0f;

	// 1. Enemy distance score
	if (EnemyDistanceWeight > 0.0f)
	{
		float AvgEnemyDistance = 0.0f;
		for (const FVector& EnemyLocation : EnemyLocations)
		{
			AvgEnemyDistance += FVector::Dist(CoverLocation, EnemyLocation);
		}
		AvgEnemyDistance /= EnemyLocations.Num();

		// Normalize between MinSafeDistance and MaxSafeDistance
		float DistanceScore = FMath::Clamp(
			(AvgEnemyDistance - MinSafeDistance) / (MaxSafeDistance - MinSafeDistance),
			0.0f,
			1.0f
		);

		TotalScore += DistanceScore * EnemyDistanceWeight;
		TotalWeight += EnemyDistanceWeight;
	}

	// 2. Line of sight score (prefer blocked LOS to enemies)
	if (bCheckLineOfSight && LineOfSightWeight > 0.0f)
	{
		int32 BlockedCount = 0;
		for (const FVector& EnemyLocation : EnemyLocations)
		{
			if (!HasLineOfSight(CoverLocation, EnemyLocation, World))
			{
				BlockedCount++;
			}
		}

		float LOSScore = static_cast<float>(BlockedCount) / EnemyLocations.Num();
		TotalScore += LOSScore * LineOfSightWeight;
		TotalWeight += LineOfSightWeight;
	}

	// 3. Distance from querier (prefer closer)
	if (QuerierDistanceWeight > 0.0f)
	{
		float QuerierDistance = FVector::Dist(CoverLocation, QuerierLocation);
		float ProximityScore = 1.0f - FMath::Clamp(QuerierDistance / MaxSafeDistance, 0.0f, 1.0f);

		TotalScore += ProximityScore * QuerierDistanceWeight;
		TotalWeight += QuerierDistanceWeight;
	}

	// Normalize by total weight
	if (TotalWeight > 0.0f)
	{
		TotalScore /= TotalWeight;
	}

	return TotalScore;
}

bool UEnvQueryTest_CoverQuality::HasLineOfSight(const FVector& From, const FVector& To, UWorld* World) const
{
	FHitResult HitResult;
	FCollisionQueryParams Params;
	Params.bTraceComplex = false;
	Params.bReturnPhysicalMaterial = false;

	// Trace at eye level (add 150cm height offset)
	FVector FromEyeLevel = From + FVector(0, 0, 150.0f);
	FVector ToEyeLevel = To + FVector(0, 0, 150.0f);

	bool bHit = World->LineTraceSingleByChannel(
		HitResult,
		FromEyeLevel,
		ToEyeLevel,
		ECC_Visibility,
		Params
	);

	// If trace hit something, LOS is blocked
	return !bHit;
}

FText UEnvQueryTest_CoverQuality::GetDescriptionTitle() const
{
	return FText::FromString(TEXT("Cover Quality"));
}

FText UEnvQueryTest_CoverQuality::GetDescriptionDetails() const
{
	return FText::FromString(FString::Printf(
		TEXT("Enemy Dist: %.2f, LOS: %.2f, Proximity: %.2f"),
		EnemyDistanceWeight,
		LineOfSightWeight,
		QuerierDistanceWeight
	));
}
