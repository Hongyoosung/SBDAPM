// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Conditions/STCondition_CheckCommandType.h"

bool FSTCondition_CheckCommandType::TestCondition(FStateTreeExecutionContext& Context) const
{
	const FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Get command data from bound properties
	const EStrategicCommandType CurrentCommandType = InstanceData.CurrentCommandType;
	const bool bIsCommandValid = InstanceData.bIsCommandValid;

	// DEBUG: Log condition evaluation (Warning level for visibility)
	UE_LOG(LogTemp, Warning, TEXT("[CHECK COMMAND TYPE] Evaluating: Valid=%d, Type=%s, Accepted=%d types, RequireValid=%d"),
		bIsCommandValid,
		*UEnum::GetValueAsString(CurrentCommandType),
		InstanceData.AcceptedCommandTypes.Num(),
		InstanceData.bRequireValidCommand);

	// Log accepted types for debugging
	if (InstanceData.AcceptedCommandTypes.Num() > 0)
	{
		FString AcceptedStr;
		for (const EStrategicCommandType& Type : InstanceData.AcceptedCommandTypes)
		{
			AcceptedStr += UEnum::GetValueAsString(Type) + TEXT(", ");
		}
		UE_LOG(LogTemp, Warning, TEXT("[CHECK COMMAND TYPE] Accepted types: [%s]"), *AcceptedStr);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("[CHECK COMMAND TYPE] No AcceptedCommandTypes configured!"));
		return InstanceData.bInvertCondition;
	}

	// Check validity if required
	if (InstanceData.bRequireValidCommand && !bIsCommandValid)
	{
		//UE_LOG(LogTemp, Warning, TEXT("[CHECK COMMAND TYPE] Command invalid (bIsCommandValid=false), condition fails"));
		return InstanceData.bInvertCondition; // Invalid command = false (or true if inverted)
	}

	// Check if command type matches any accepted types
	bool bMatches = InstanceData.AcceptedCommandTypes.Contains(CurrentCommandType);

	bool bResult = InstanceData.bInvertCondition ? !bMatches : bMatches;
	UE_LOG(LogTemp, Warning, TEXT("[CHECK COMMAND TYPE] %s (CommandType=%s, Matches=%d, Inverted=%d) -> Result=%d"),
		bResult ? TEXT("PASS") : TEXT("FAIL"),
		*UEnum::GetValueAsString(CurrentCommandType),
		bMatches,
		InstanceData.bInvertCondition,
		bResult);

	return bResult;
}
