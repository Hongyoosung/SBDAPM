#include "EQS/EnvQueryGenerator_CoverPoints.h"
#include "EnvironmentQuery/Contexts/EnvQueryContext_Querier.h"
#include "EnvironmentQuery/Items/EnvQueryItemType_Point.h"
#include "Kismet/GameplayStatics.h"

UEnvQueryGenerator_CoverPoints::UEnvQueryGenerator_CoverPoints()
{
	ItemType = UEnvQueryItemType_Point::StaticClass();
	SearchRadius = 1500.0f;
	GridSpacing = 200.0f;
	bIncludeTaggedCover = true;
	bGenerateGridPoints = true;
	GroundTraceDistance = 500.0f;
}

void UEnvQueryGenerator_CoverPoints::GenerateItems(FEnvQueryInstance& QueryInstance) const
{
	TArray<FVector> GeneratedPoints;

	// Get querier location
	TArray<FVector> QuerierLocations;
	QueryInstance.PrepareContext(UEnvQueryContext_Querier::StaticClass(), QuerierLocations);

	if (QuerierLocations.Num() == 0)
	{
		return;
	}

	const FVector& QuerierLocation = QuerierLocations[0];
	UWorld* World = QueryInstance.World;

	// 1. Include pre-placed cover actors with tags
	if (bIncludeTaggedCover)
	{
		TArray<AActor*> CoverActors;
		UGameplayStatics::GetAllActorsWithTag(World, CoverTag, CoverActors);

		for (AActor* CoverActor : CoverActors)
		{
			if (CoverActor)
			{
				FVector CoverLocation = CoverActor->GetActorLocation();
				float Distance = FVector::Dist(QuerierLocation, CoverLocation);

				if (Distance <= SearchRadius)
				{
					GeneratedPoints.Add(CoverLocation);
				}
			}
		}
	}

	// 2. Generate grid-based sample points
	if (bGenerateGridPoints)
	{
		const int32 GridSteps = FMath::CeilToInt(SearchRadius / GridSpacing);

		for (int32 X = -GridSteps; X <= GridSteps; ++X)
		{
			for (int32 Y = -GridSteps; Y <= GridSteps; ++Y)
			{
				FVector Offset(X * GridSpacing, Y * GridSpacing, 0.0f);
				FVector TestPoint = QuerierLocation + Offset;

				// Check if within radius
				if (Offset.Size2D() <= SearchRadius)
				{
					// Trace down to ground
					FHitResult HitResult;
					FVector TraceStart = TestPoint + FVector(0, 0, 100.0f);
					FVector TraceEnd = TestPoint - FVector(0, 0, GroundTraceDistance);

					FCollisionQueryParams TraceParams;
					TraceParams.bTraceComplex = false;

					if (World->LineTraceSingleByChannel(HitResult, TraceStart, TraceEnd, ECC_WorldStatic, TraceParams))
					{
						GeneratedPoints.Add(HitResult.Location);
					}
					else
					{
						// Use original point if no ground found
						GeneratedPoints.Add(TestPoint);
					}
				}
			}
		}
	}

	// Add all generated points to the query
	for (const FVector& Point : GeneratedPoints)
	{
		QueryInstance.AddItemData<UEnvQueryItemType_Point>(Point);
	}
}

FText UEnvQueryGenerator_CoverPoints::GetDescriptionTitle() const
{
	return FText::FromString(FString::Printf(TEXT("Cover Points: Radius %.0f"), SearchRadius));
}

FText UEnvQueryGenerator_CoverPoints::GetDescriptionDetails() const
{
	return FText::FromString(FString::Printf(
		TEXT("Grid spacing: %.0f, Tag: %s, Include tagged: %s, Generate grid: %s"),
		GridSpacing,
		*CoverTag.ToString(),
		bIncludeTaggedCover ? TEXT("Yes") : TEXT("No"),
		bGenerateGridPoints ? TEXT("Yes") : TEXT("No")
	));
}
