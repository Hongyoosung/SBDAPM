#include "Utill/GameAIHelper.h"
#include "Combat/HealthComponent.h" // HealthComponent 헤더 경로 확인 필요
#include "GameFramework/Actor.h"
#include "GameFramework/Pawn.h"

bool UGameAIHelper::IsTargetValid(AActor* Target)
{
	if (!Target)
	{
		UE_LOG(LogTemp, Warning, TEXT("[GameAIHelper] IsTargetValid: Target is nullptr"));
		return false;
	}

	if (!Target->IsValidLowLevel())
	{
		UE_LOG(LogTemp, Warning, TEXT("[GameAIHelper] IsTargetValid: '%s' is NOT valid low level"), *Target->GetName());
		return false;
	}

	if (Target->IsPendingKillPending())
	{
		UE_LOG(LogTemp, Warning, TEXT("[GameAIHelper] IsTargetValid: '%s' is pending kill"), *Target->GetName());
		return false;
	}

	// 타겟의 HealthComponent 확인 (살아있는지)
	if (UHealthComponent* HealthComp = Target->FindComponentByClass<UHealthComponent>())
	{
		bool bIsAlive = HealthComp->IsAlive();
		if (!bIsAlive)
		{
			UE_LOG(LogTemp, Warning, TEXT("[GameAIHelper] IsTargetValid: '%s' is DEAD (Health=%.1f)"),
				*Target->GetName(), HealthComp->GetCurrentHealth());
		}
		return bIsAlive;
	}

	// No health component - target is valid
	return true;
}

AActor* UGameAIHelper::FindNearestValidEnemy(const TArray<AActor*>& VisibleEnemies, APawn* FromPawn)
{
	if (!FromPawn) return nullptr;

	UE_LOG(LogTemp, Display, TEXT("[GameAIHelper] FindNearestValidEnemy: Searching %d visible enemies for '%s'"),
		VisibleEnemies.Num(), *FromPawn->GetName());

	FVector MyLocation = FromPawn->GetActorLocation();
	AActor* NearestEnemy = nullptr;
	float NearestDistance = FLT_MAX;

	for (AActor* Enemy : VisibleEnemies)
	{
		// 위에서 만든 static 함수 호출
		if (IsTargetValid(Enemy))
		{
			float Distance = FVector::Dist(MyLocation, Enemy->GetActorLocation());
			if (Distance < NearestDistance)
			{
				NearestDistance = Distance;
				NearestEnemy = Enemy;
			}
		}
	}

	if (NearestEnemy)
	{
		UE_LOG(LogTemp, Display, TEXT("[GameAIHelper] FindNearestValidEnemy: Found '%s' at distance %.1f"),
			*NearestEnemy->GetName(), NearestDistance);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("[GameAIHelper] FindNearestValidEnemy: NO valid enemies found!"));
	}

	return NearestEnemy;
}