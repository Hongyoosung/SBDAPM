// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/FollowerStateTreeSchema.h"
#include "StateTreeExecutionContext.h"
#include "AIController.h"
#include "Team/FollowerAgentComponent.h"
#include "Team/TeamLeaderComponent.h"
#include "RL/RLPolicyNetwork.h"

TArray<FStateTreeExternalDataDesc> UFollowerStateTreeSchema::ContextDataDescs;

UFollowerStateTreeSchema::UFollowerStateTreeSchema()
{
	// Bind the FFollowerStateTreeContext as the context data for this schema
	ContextDataDescs.Reset();

	FStateTreeExternalDataDesc& ContextDesc = ContextDataDescs.AddDefaulted_GetRef();
	ContextDesc.Struct = FFollowerStateTreeContext::StaticStruct();
	ContextDesc.Name = FName(TEXT("FollowerContext"));
	ContextDesc.ID = FGuid::NewGuid();
	ContextDesc.Requirement = EStateTreeExternalDataRequirement::Required;
}

bool UFollowerStateTreeSchema::IsStructAllowed(const UScriptStruct* InScriptStruct) const
{
	// Allow base State Tree structs
	if (Super::IsStructAllowed(InScriptStruct))
	{
		return true;
	}

	// Allow project-specific structs
	if (InScriptStruct)
	{
		// Allow observation structs
		if (InScriptStruct->GetFName() == FName(TEXT("ObservationElement")))
		{
			return true;
		}

		// Allow RL types
		if (InScriptStruct->GetFName() == FName(TEXT("TacticalAction")) ||
			InScriptStruct->GetFName() == FName(TEXT("StrategicCommand")))
		{
			return true;
		}

		// Allow team types
		if (InScriptStruct->GetFName() == FName(TEXT("TeamID")) ||
			InScriptStruct->GetFName() == FName(TEXT("TeamMessage")))
		{
			return true;
		}
	}

	return false;
}

bool UFollowerStateTreeSchema::IsClassAllowed(const UClass* InClass) const
{
	// Allow base State Tree classes
	if (Super::IsClassAllowed(InClass))
	{
		return true;
	}

	// Allow AI-related classes
	if (InClass)
	{
		if (InClass->IsChildOf(AAIController::StaticClass()) ||
			InClass->IsChildOf(UFollowerAgentComponent::StaticClass()) ||
			InClass->IsChildOf(UTeamLeaderComponent::StaticClass()) ||
			InClass->IsChildOf(URLPolicyNetwork::StaticClass()))
		{
			return true;
		}

		// Allow Actor and APawn for target tracking
		if (InClass->IsChildOf(AActor::StaticClass()))
		{
			return true;
		}
	}

	return false;
}

bool UFollowerStateTreeSchema::IsExternalItemAllowed(const UStruct& InStruct) const
{
	// Allow base State Tree external items
	if (Super::IsExternalItemAllowed(InStruct))
	{
		return true;
	}

	// Allow our context struct
	if (&InStruct == FFollowerStateTreeContext::StaticStruct())
	{
		return true;
	}

	return false;
}

TConstArrayView<FStateTreeExternalDataDesc> UFollowerStateTreeSchema::GetContextDataDescs() const
{
	return ContextDataDescs;
}
