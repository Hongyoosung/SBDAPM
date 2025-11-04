#include "EQS/EnvQueryContext_CoverEnemies.h"
#include "EnvironmentQuery/EnvQueryTypes.h"
#include "EnvironmentQuery/Items/EnvQueryItemType_Point.h"
#include "AIController.h"
#include "Team/TeamLeaderComponent.h"
#include "Team/FollowerAgentComponent.h"
#include "GameFramework/Pawn.h"

void UEnvQueryContext_CoverEnemies::ProvideContext(FEnvQueryInstance& QueryInstance, FEnvQueryContextData& ContextData) const
{
	AActor* QueryOwner = Cast<AActor>(QueryInstance.Owner.Get());
	if (!QueryOwner)
	{
		return;
	}

	TArray<FVector> EnemyLocations;

	// Try to get team leader component
	UTeamLeaderComponent* TeamLeader = QueryOwner->FindComponentByClass<UTeamLeaderComponent>();
	if (TeamLeader)
	{
		// Get known enemies from team leader
		TArray<AActor*> Enemies = TeamLeader->GetKnownEnemies();
		for (AActor* Enemy : Enemies)
		{
			if (Enemy)
			{
				EnemyLocations.Add(Enemy->GetActorLocation());
			}
		}
	}
	else
	{
		// Fallback: Try follower component
		UFollowerAgentComponent* Follower = QueryOwner->FindComponentByClass<UFollowerAgentComponent>();
		if (Follower)
		{
			UTeamLeaderComponent* Leader = Follower->GetTeamLeader();
			if (Leader)
			{
				TArray<AActor*> Enemies = Leader->GetKnownEnemies();
				for (AActor* Enemy : Enemies)
				{
					if (Enemy)
					{
						EnemyLocations.Add(Enemy->GetActorLocation());
					}
				}
			}
		}
	}

	// Set enemy locations as context data
	if (EnemyLocations.Num() > 0)
	{
		UEnvQueryItemType_Point::SetContextHelper(ContextData, EnemyLocations);
	}
}
