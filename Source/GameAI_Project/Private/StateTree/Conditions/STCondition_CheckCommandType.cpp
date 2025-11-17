// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Conditions/STCondition_CheckCommandType.h"
#include "StateTree/FollowerStateTreeContext.h"

bool FSTCondition_CheckCommandType::TestCondition(FStateTreeExecutionContext& Context) const
{
	const FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// DEBUG: Log condition evaluation
	UE_LOG(LogTemp, Verbose, TEXT("[CHECK COMMAND] Valid=%d, Type=%s, Accepted=%d types"),
		InstanceData.bIsCommandValid,
		*UEnum::GetValueAsString(InstanceData.CurrentCommand.CommandType),
		InstanceData.AcceptedCommandTypes.Num());

	// Check validity if required
	if (InstanceData.bRequireValidCommand && !InstanceData.bIsCommandValid)
	{
		UE_LOG(LogTemp, Warning, TEXT("[CHECK COMMAND] ❌ Command invalid, condition fails"));
		return InstanceData.bInvertCondition; // Invalid command = false (or true if inverted)
	}

	// Check if command type matches any accepted types
	bool bMatches = InstanceData.AcceptedCommandTypes.Contains(InstanceData.CurrentCommand.CommandType);

	UE_LOG(LogTemp, Display, TEXT("[CHECK COMMAND] %s (CommandType=%s, Matches=%d, Inverted=%d)"),
		(InstanceData.bInvertCondition ? !bMatches : bMatches) ? TEXT("✅ PASS") : TEXT("❌ FAIL"),
		*UEnum::GetValueAsString(InstanceData.CurrentCommand.CommandType),
		bMatches,
		InstanceData.bInvertCondition);

	// Apply inversion if needed
	return InstanceData.bInvertCondition ? !bMatches : bMatches;
}
