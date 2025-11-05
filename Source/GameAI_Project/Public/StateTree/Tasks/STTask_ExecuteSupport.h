// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeTaskBase.h"
#include "RL/RLTypes.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "STTask_ExecuteSupport.generated.h"


/**
 * State Tree Task: Execute Support
 *
 * Executes support tactics:
 * - Provide covering fire
 * - Suppressive fire
 * - Reload
 * - Use abilities
 */


USTRUCT()
struct GAMEAI_PROJECT_API FSTTask_ExecuteSupportInstanceData
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, Category = "Context")
	FFollowerStateTreeContext Context;

	UPROPERTY(EditAnywhere, Category = "Config")
	float RLQueryInterval = 3.0f;

	UPROPERTY()
	float TimeSinceLastRLQuery = 0.0f;
};

USTRUCT(meta = (DisplayName = "Execute Support"))
struct GAMEAI_PROJECT_API FSTTask_ExecuteSupport : public FStateTreeTaskBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTTask_ExecuteSupportInstanceData;

	FSTTask_ExecuteSupport() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }

	virtual EStateTreeRunStatus EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const override;
	virtual EStateTreeRunStatus Tick(FStateTreeExecutionContext& Context, float DeltaTime) const override;
	virtual void ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const override;

protected:
	void ExecuteTacticalAction(FStateTreeExecutionContext& Context, float DeltaTime) const;
	void ExecuteProvideCoveringFire(FStateTreeExecutionContext& Context, float DeltaTime) const;
	void ExecuteSuppressiveFire(FStateTreeExecutionContext& Context, float DeltaTime) const;
	void ExecuteReload(FStateTreeExecutionContext& Context, float DeltaTime) const;
	void ExecuteUseAbility(FStateTreeExecutionContext& Context, float DeltaTime) const;
};