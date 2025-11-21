#include "Util/GameAIHelper.h"
#include "Combat/HealthComponent.h"
#include "Team/FollowerAgentComponent.h"
#include "Team/TeamLeaderComponent.h"
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

FVector UGameAIHelper::CalculateFormationOffset(
	AActor* Agent,
	UFollowerAgentComponent* FollowerComponent,
	EStrategicCommandType CommandType)
{
	if (!Agent || !FollowerComponent || !FollowerComponent->TeamLeader)
	{
		return FVector::ZeroVector;
	}

	// Get team members to determine agent's position in formation
	TArray<AActor*> TeamMembers = FollowerComponent->TeamLeader->GetAliveFollowers();
	int32 AgentIndex = TeamMembers.IndexOfByKey(Agent);
	int32 TeamSize = TeamMembers.Num();

	if (AgentIndex == INDEX_NONE || TeamSize <= 1)
	{
		// Single agent or not found - no offset needed
		return FVector::ZeroVector;
	}

	// Base offset distance (500cm = optimal spacing center)
	const float OffsetDistance = 500.0f;

	// Calculate normalized position in team (-1.0 to +1.0)
	// For 3 agents: indices 0,1,2 → positions -1, 0, +1
	float NormalizedPosition = (AgentIndex - (TeamSize - 1) * 0.5f);

	FVector Offset = FVector::ZeroVector;

	switch (CommandType)
	{
	case EStrategicCommandType::Assault:
		// Assault: Horizontal line formation (spread left/right of target)
		// Agent 0 → left, Agent 1 → center, Agent 2 → right
		Offset = FVector(0.0f, NormalizedPosition * OffsetDistance, 0.0f);
		break;

	case EStrategicCommandType::Support:
		// Support: Stay back with slight horizontal spread
		// Further from target to provide covering fire
		Offset = FVector(-OffsetDistance * 0.8f, NormalizedPosition * OffsetDistance * 0.5f, 0.0f);
		break;

	case EStrategicCommandType::TakeCover:
	case EStrategicCommandType::HoldPosition:
		// Defend: Dispersed formation (wider spread, slightly back)
		Offset = FVector(-OffsetDistance * 0.3f, NormalizedPosition * OffsetDistance * 0.9f, 0.0f);
		break;

	case EStrategicCommandType::Advance:
	case EStrategicCommandType::MoveTo:
		// Movement: Staggered column (slightly offset front/back and left/right)
		Offset = FVector(NormalizedPosition * OffsetDistance * 0.3f, NormalizedPosition * OffsetDistance * 0.6f, 0.0f);
		break;

	case EStrategicCommandType::Retreat:
		// Retreat: Spread out to avoid area damage during withdrawal
		Offset = FVector(NormalizedPosition * OffsetDistance * 0.4f, NormalizedPosition * OffsetDistance * 0.8f, 0.0f);
		break;

	default:
		// Default: Simple horizontal spread
		Offset = FVector(0.0f, NormalizedPosition * OffsetDistance, 0.0f);
		break;
	}

	UE_LOG(LogTemp, Log, TEXT("[FORMATION OFFSET] Agent '%s' [%d/%d]: Command=%s, Offset=%s"),
		*Agent->GetName(),
		AgentIndex + 1,
		TeamSize,
		*UEnum::GetValueAsString(CommandType),
		*Offset.ToString());

	return Offset;
}