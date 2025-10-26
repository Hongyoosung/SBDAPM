// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/BTDecorator_CheckStrategy.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "BehaviorTree/BehaviorTreeComponent.h"

UBTDecorator_CheckStrategy::UBTDecorator_CheckStrategy()
{
	NodeName = "Check Strategy";

	// This decorator checks a condition, so we want it to be reevaluated
	// when the Blackboard value changes
	bNotifyBecomeRelevant = true;
	bNotifyTick = false; // We don't need tick, just notification when becoming relevant
}

bool UBTDecorator_CheckStrategy::CalculateRawConditionValue(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) const
{
	// Get the Blackboard component
	UBlackboardComponent* BlackboardComp = OwnerComp.GetBlackboardComponent();

	if (!BlackboardComp)
	{
		if (bEnableDebugLog)
		{
			UE_LOG(LogTemp, Warning, TEXT("BTDecorator_CheckStrategy: No Blackboard component found"));
		}
		return false;
	}

	// Get the current strategy from the Blackboard
	FString CurrentStrategy = BlackboardComp->GetValueAsString(StrategyKeyName);

	// Check if it matches the required strategy
	bool bMatches = CurrentStrategy.Equals(RequiredStrategy, ESearchCase::IgnoreCase);

	if (bEnableDebugLog)
	{
		if (bMatches)
		{
			UE_LOG(LogTemp, Log, TEXT("BTDecorator_CheckStrategy: Strategy MATCH - Current: '%s', Required: '%s'"),
				*CurrentStrategy, *RequiredStrategy);
		}
		else
		{
			UE_LOG(LogTemp, Log, TEXT("BTDecorator_CheckStrategy: Strategy MISMATCH - Current: '%s', Required: '%s'"),
				*CurrentStrategy, *RequiredStrategy);
		}
	}

	return bMatches;
}

FString UBTDecorator_CheckStrategy::GetStaticDescription() const
{
	return FString::Printf(TEXT("Check if strategy is '%s'"), *RequiredStrategy);
}
