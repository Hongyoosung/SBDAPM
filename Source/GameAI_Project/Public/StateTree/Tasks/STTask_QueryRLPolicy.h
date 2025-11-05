// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeTaskBase.h"
#include "RL/RLTypes.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "STTask_QueryRLPolicy.generated.h"


/**
 * State Tree Task: Query RL Policy
 *
 * Queries the follower's RL policy network for the next tactical action
 * and updates the context with the selected action.
 *
 * This task runs instantly and completes immediately after querying the policy.
 * It should be used as a decision point before tactical execution tasks.
 *
 * Requirements:
 * - FollowerStateTreeContext with valid TacticalPolicy
 * - CurrentObservation must be up-to-date (handled by evaluator)
 *
 * Output:
 * - Updates CurrentTacticalAction in context
 * - Resets TimeInTacticalAction to 0
 *
 * Usage:
 * - Place before Execute tasks to select which tactic to use
 * - Can be run periodically or only at state entry
 */


/**
 * Instance data for QueryRLPolicy task
 */
USTRUCT()
struct GAMEAI_PROJECT_API FSTTask_QueryRLPolicyInstanceData
{
	GENERATED_BODY()

	/** Follower component (bound from context) */
	UPROPERTY(EditAnywhere, Category = "Input")
	TObjectPtr<UFollowerAgentComponent> FollowerComponent = nullptr;

	/** RL Policy network (bound from context) */
	UPROPERTY(EditAnywhere, Category = "Input")
	TObjectPtr<URLPolicyNetwork> TacticalPolicy = nullptr;

	/** Current observation (bound from context) */
	UPROPERTY(EditAnywhere, Category = "Input")
	FObservationElement CurrentObservation;

	/** Current command (bound from context) */
	UPROPERTY(EditAnywhere, Category = "Input")
	FStrategicCommand CurrentCommand;

	/** In cover flag (bound from context) */
	UPROPERTY(EditAnywhere, Category = "Input")
	bool bInCover = false;

	/** Selected tactical action (output - bound to context) */
	UPROPERTY(EditAnywhere, Category = "Output")
	ETacticalAction SelectedAction = ETacticalAction::DefensiveHold;

	/** Log action selection to console */
	UPROPERTY(EditAnywhere, Category = "Config")
	bool bLogActionSelection = true;

	/** Draw debug visualization */
	UPROPERTY(EditAnywhere, Category = "Config")
	bool bDrawDebugInfo = false;

	/** Use RL policy (if false, uses rule-based fallback) */
	UPROPERTY(EditAnywhere, Category = "Config")
	bool bUseRLPolicy = true;
};

USTRUCT(meta = (DisplayName = "Query RL Policy"))
struct GAMEAI_PROJECT_API FSTTask_QueryRLPolicy : public FStateTreeTaskBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTTask_QueryRLPolicyInstanceData;

	FSTTask_QueryRLPolicy() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }

	virtual EStateTreeRunStatus EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const override;
	virtual void ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const override;

protected:
	/** Query the RL policy and update context */
	ETacticalAction QueryPolicy(FStateTreeExecutionContext& Context) const;

	/** Get rule-based fallback action */
	ETacticalAction GetFallbackAction(FStateTreeExecutionContext& Context) const;
};
