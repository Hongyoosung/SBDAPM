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
	// [중요] TConstArrayView를 리턴해야 하므로, 데이터는 함수가 끝나도 메모리에 남아있어야 합니다.
	// static을 사용하여 한 번만 초기화하고 메모리를 유지합니다.
	static TArray<FStateTreeExternalDataDesc> CachedDescs;

	if (CachedDescs.Num() == 0)
	{
		// 1. 부모 클래스(UStateTreeComponentSchema)의 데이터 가져오기 (Pawn, AIController 등)
		TConstArrayView<FStateTreeExternalDataDesc> ParentDescs = Super::GetContextDataDescs();
		CachedDescs.Append(ParentDescs.GetData(), ParentDescs.Num());

		// 2. 커스텀 데이터 추가 (반드시 고정된 GUID 사용)

		// (1) Follower Context (Struct)
		FStateTreeExternalDataDesc ContextDesc;
		ContextDesc.Name = FName(TEXT("FollowerContext"));
		ContextDesc.Struct = FFollowerStateTreeContext::StaticStruct();
		ContextDesc.Requirement = EStateTreeExternalDataRequirement::Required;
		// 절대 변경하지 말 것 (기존 에셋과의 연결 고리)
		ContextDesc.ID = FGuid(0xA1B2C3D4, 0xE5F60001, 0x11223344, 0x55667788);
		CachedDescs.Add(ContextDesc);

		// (2) Follower Agent Component
		FStateTreeExternalDataDesc FollowerDesc;
		FollowerDesc.Name = FName(TEXT("FollowerComponent"));
		FollowerDesc.Struct = UFollowerAgentComponent::StaticClass();
		FollowerDesc.Requirement = EStateTreeExternalDataRequirement::Required;
		ContextDesc.ID = FGuid(0xA1B2C3D4, 0xE5F60002, 0x11223344, 0x55667788);
		CachedDescs.Add(FollowerDesc);

		// (3) Follower State Tree Component (Self)
		FStateTreeExternalDataDesc SelfDesc;
		SelfDesc.Name = FName(TEXT("FollowerStateTreeComponent"));
		SelfDesc.Struct = UFollowerStateTreeComponent::StaticClass();
		SelfDesc.Requirement = EStateTreeExternalDataRequirement::Required;
		SelfDesc.ID = FGuid(0xA1B2C3D4, 0xE5F60005, 0x11223344, 0x55667788);
		CachedDescs.Add(SelfDesc);

		// (4) Team Leader (Optional)
		FStateTreeExternalDataDesc LeaderDesc;
		LeaderDesc.Name = FName(TEXT("TeamLeader"));
		LeaderDesc.Struct = UTeamLeaderComponent::StaticClass();
		LeaderDesc.Requirement = EStateTreeExternalDataRequirement::Optional;
		LeaderDesc.ID = FGuid(0xA1B2C3D4, 0xE5F60003, 0x11223344, 0x55667788);
		CachedDescs.Add(LeaderDesc);

		// (5) Tactical Policy (Optional)
		FStateTreeExternalDataDesc PolicyDesc;
		PolicyDesc.Name = FName(TEXT("TacticalPolicy"));
		PolicyDesc.Struct = URLPolicyNetwork::StaticClass();
		PolicyDesc.Requirement = EStateTreeExternalDataRequirement::Optional;
		PolicyDesc.ID = FGuid(0xA1B2C3D4, 0xE5F60004, 0x11223344, 0x55667788);
		CachedDescs.Add(PolicyDesc);
	}

	return CachedDescs;
}

void UFollowerStateTreeSchema::SetContextData(FContextDataSetter& ContextDataSetter, bool bLogErrors) const
{
	// Call parent to handle all descriptors
	// Parent's implementation handles UActorComponent types via FindComponentByClass
	// This will automatically find: FollowerComponent, TeamLeader, TacticalPolicy
	Super::SetContextData(ContextDataSetter, bLogErrors);
}