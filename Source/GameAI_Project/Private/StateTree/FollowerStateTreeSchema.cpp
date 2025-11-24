// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/FollowerStateTreeSchema.h"
#include "StateTree/FollowerStateTreeComponent.h"
#include "StateTreeExecutionContext.h"
#include "Components/StateTreeComponent.h"
#include "AI/AIController/FollowerAIController.h"
#include "Actor/FollowerCharacter.h"
#include "Team/FollowerAgentComponent.h"
#include "Team/TeamLeaderComponent.h"
#include "RL/RLPolicyNetwork.h"
#include "StateTree/Conditions/STCondition_IsAlive.h"
#include "StateTree/Conditions/STCondition_CheckCommandType.h"
#include "StateTree/Conditions/STCondition_CheckTacticalAction.h"
#include "StateTree/Tasks/STTask_ExecuteAssault.h"
#include "StateTree/Tasks/STTask_ExecuteDefend.h"
#include "StateTree/Tasks/STTask_ExecuteSupport.h"
#include "StateTree/Tasks/STTask_ExecuteMove.h"
#include "StateTree/Tasks/STTask_ExecuteRetreat.h"
#include "StateTree/Tasks/STTask_Dead.h"
#include "StateTree/Tasks/STTask_QueryRLPolicy.h"
#include "StateTree/Evaluators/STEvaluator_SyncCommand.h"
#include "StateTree/Evaluators/STEvaluator_UpdateObservation.h"


UFollowerStateTreeSchema::UFollowerStateTreeSchema()
{
	AIControllerClass = AFollowerAIController::StaticClass();
	PawnClass = AFollowerCharacter::StaticClass();
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
		// Allow all StateTree node types
		if (InScriptStruct->IsChildOf(FSTEvaluator_SyncCommand::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTEvaluator_UpdateObservation::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTCondition_IsAlive::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTCondition_CheckCommandType::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTCondition_CheckTacticalAction::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTTask_ExecuteAssault::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTTask_ExecuteDefend::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTTask_ExecuteSupport::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTTask_QueryRLPolicy::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTTask_ExecuteMove::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTTask_ExecuteRetreat::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTTask_Dead::StaticStruct()))
		{
			return true;
		}

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

		// Allow context struct
		if (InScriptStruct == FFollowerStateTreeContext::StaticStruct())
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
			InClass->IsChildOf(APawn::StaticClass()) ||
			InClass->IsChildOf(UFollowerAgentComponent::StaticClass()) ||
			InClass->IsChildOf(UTeamLeaderComponent::StaticClass()) ||
			InClass->IsChildOf(URLPolicyNetwork::StaticClass()) ||
			InClass->IsChildOf(UFollowerStateTreeComponent::StaticClass()))
		{
			return true;
		}

		// Allow Actor for target tracking
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
	// [수정됨] 이미 데이터가 구축되어 있다면 캐시된 데이터를 즉시 반환합니다.
	// 이것이 에셋 안정성을 보장하는 핵심입니다.
	if (ContextDataDescs.Num() > 0)
	{
		return ContextDataDescs;
	}

	// 1. 부모 클래스 데이터 가져오기 (AIController, Pawn 등)
	TConstArrayView<FStateTreeExternalDataDesc> ParentDescs = Super::GetContextDataDescs();
	ContextDataDescs.Append(ParentDescs.GetData(), ParentDescs.Num());

	// 2. 커스텀 컨텍스트 데이터 추가
	// (GUID는 기존과 동일하게 유지하여 에셋 호환성 확보)

	// (1) Follower Context
	{
		FStateTreeExternalDataDesc Desc;
		Desc.Name = FName(TEXT("FollowerContext"));
		Desc.Struct = FFollowerStateTreeContext::StaticStruct();
		Desc.Requirement = EStateTreeExternalDataRequirement::Required;
		Desc.ID = FGuid(0xA1B2C3D4, 0xE5F60001, 0x11223344, 0x55667788);
		ContextDataDescs.Add(Desc);
	}

	// (2) Follower Agent Component
	{
		FStateTreeExternalDataDesc Desc;
		Desc.Name = FName(TEXT("FollowerComponent"));
		Desc.Struct = UFollowerAgentComponent::StaticClass();
		Desc.Requirement = EStateTreeExternalDataRequirement::Required;
		Desc.ID = FGuid(0xA1B2C3D4, 0xE5F60002, 0x11223344, 0x55667788);
		ContextDataDescs.Add(Desc);
	}

	// (3) Follower State Tree Component
	{
		FStateTreeExternalDataDesc Desc;
		Desc.Name = FName(TEXT("FollowerStateTreeComponent"));
		Desc.Struct = UFollowerStateTreeComponent::StaticClass();
		Desc.Requirement = EStateTreeExternalDataRequirement::Required;
		Desc.ID = FGuid(0xA1B2C3D4, 0xE5F60005, 0x11223344, 0x55667788);
		ContextDataDescs.Add(Desc);
	}

	// (4) Team Leader
	{
		FStateTreeExternalDataDesc Desc;
		Desc.Name = FName(TEXT("TeamLeader"));
		Desc.Struct = UTeamLeaderComponent::StaticClass();
		Desc.Requirement = EStateTreeExternalDataRequirement::Optional;
		Desc.ID = FGuid(0xA1B2C3D4, 0xE5F60003, 0x11223344, 0x55667788);
		ContextDataDescs.Add(Desc);
	}

	// (5) Tactical Policy
	{
		FStateTreeExternalDataDesc Desc;
		Desc.Name = FName(TEXT("TacticalPolicy"));
		Desc.Struct = URLPolicyNetwork::StaticClass();
		Desc.Requirement = EStateTreeExternalDataRequirement::Optional;
		Desc.ID = FGuid(0xA1B2C3D4, 0xE5F60004, 0x11223344, 0x55667788);
		ContextDataDescs.Add(Desc);
	}

	return ContextDataDescs;
}

void UFollowerStateTreeSchema::SetContextData(FContextDataSetter& ContextDataSetter, bool bLogErrors) const
{
	// Call parent to handle all descriptors
	// Parent's implementation handles UActorComponent types via FindComponentByClass
	// This will automatically find: FollowerComponent, TeamLeader, TacticalPolicy
	Super::SetContextData(ContextDataSetter, bLogErrors);
}