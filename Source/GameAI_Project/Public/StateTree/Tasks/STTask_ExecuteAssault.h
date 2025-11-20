// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeTaskBase.h"
#include "RL/RLTypes.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "StateTreeExecutionTypes.h"
#include "STTask_ExecuteAssault.generated.h"

class UFollowerStateTreeComponent;


/**
 * State Tree Task: Execute Assault
 *
 * Executes assault tactics based on RL policy selection.
 * Handles aggressive offensive maneuvers including:
 * - Aggressive assault (full speed advance + fire)
 * - Cautious advance (slower, use cover while advancing)
 * - Flanking maneuvers (left/right)
 * - Maintain distance (kite while firing)
 *
 * Requirements:
 * - CurrentCommand.CommandType must be Assault
 * - CurrentTacticalAction set by QueryRLPolicy task
 */


USTRUCT()
struct GAMEAI_PROJECT_API FSTTask_ExecuteAssaultInstanceData
{
	GENERATED_BODY()

	/** StateTree component reference - provides access to shared context */
	UPROPERTY(EditAnywhere, Category = "Input")
	TObjectPtr<UFollowerStateTreeComponent> StateTreeComp;

	UPROPERTY(EditAnywhere, Category = "Parameter", meta = (ClampMin = "1.0", ClampMax = "2.0"))
	float AggressiveSpeedMultiplier = 1.5f;

	UPROPERTY(EditAnywhere, Category = "Parameter", meta = (ClampMin = "500.0", ClampMax = "3000.0"))
	float OptimalEngagementRange = 1500.0f;
};

USTRUCT(meta = (DisplayName = "Execute Assault"))
struct GAMEAI_PROJECT_API FSTTask_ExecuteAssault : public FStateTreeTaskBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTTask_ExecuteAssaultInstanceData;

	FSTTask_ExecuteAssault() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }

	virtual EStateTreeRunStatus EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const override;
	virtual EStateTreeRunStatus Tick(FStateTreeExecutionContext& Context, float DeltaTime) const override;
	virtual void ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const override;

protected:
	void ExecuteTacticalAction(FStateTreeExecutionContext& Context, float DeltaTime) const;
	void ExecuteAggressiveAssault(FStateTreeExecutionContext& Context, float DeltaTime) const;
	void ExecuteCautiousAdvance(FStateTreeExecutionContext& Context, float DeltaTime) const;
	void ExecuteFlankManeuver(FStateTreeExecutionContext& Context, float DeltaTime, bool bFlankLeft) const;
	void ExecuteMaintainDistance(FStateTreeExecutionContext& Context, float DeltaTime) const;
	void ExecuteDefensiveHold(FStateTreeExecutionContext& Context, float DeltaTime) const;
	void ExecuteSeekCover(FStateTreeExecutionContext& Context, float DeltaTime) const;
	void ExecuteTacticalRetreat(FStateTreeExecutionContext& Context, float DeltaTime) const;
	void ExecuteSuppressiveFire(FStateTreeExecutionContext& Context, float DeltaTime) const;
	float CalculateAssaultReward(FStateTreeExecutionContext& Context, float DeltaTime) const;
};