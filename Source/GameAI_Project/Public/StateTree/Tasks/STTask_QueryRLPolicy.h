// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeTaskBase.h"
#include "RL/RLTypes.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "StateTreeExecutionTypes.h"
#include "STTask_QueryRLPolicy.generated.h"

class UFollowerStateTreeComponent;


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

	//--------------------------------------------------------------------------
	// EXTERNAL DATA BINDING (UE 5.6 - binds to StateTree component)
	//--------------------------------------------------------------------------

	/** StateTree component reference - provides access to shared context */
	UPROPERTY(EditAnywhere, Category = "Input")
	TObjectPtr<UFollowerStateTreeComponent> StateTreeComp;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Log action selection to console */
	UPROPERTY(EditAnywhere, Category = "Parameter")
	bool bLogActionSelection = true;

	/** Draw debug visualization */
	UPROPERTY(EditAnywhere, Category = "Parameter")
	bool bDrawDebugInfo = false;

	/** Use RL policy (if false, uses rule-based fallback) */
	UPROPERTY(EditAnywhere, Category = "Parameter")
	bool bUseRLPolicy = true;

	/** Interval between policy queries (seconds). Set to 0 for one-shot query. */
	UPROPERTY(EditAnywhere, Category = "Parameter")
	float QueryInterval = 0.5f;

	/** Time since last query */
	UPROPERTY()
	float TimeSinceLastQuery = 0.0f;

	/** Has queried at least once this state entry */
	UPROPERTY()
	bool bHasQueriedOnce = false;
};

USTRUCT(meta = (DisplayName = "Query RL Policy"))
struct GAMEAI_PROJECT_API FSTTask_QueryRLPolicy : public FStateTreeTaskBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTTask_QueryRLPolicyInstanceData;

	FSTTask_QueryRLPolicy() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }

	virtual EStateTreeRunStatus EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const override;
	virtual EStateTreeRunStatus Tick(FStateTreeExecutionContext& Context, const float DeltaTime) const override;
	virtual void ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const override;

protected:
	/** Query the RL policy and update context */
	ETacticalAction QueryPolicy(FStateTreeExecutionContext& Context) const;

	/** Get rule-based fallback action */
	ETacticalAction GetFallbackAction(FStateTreeExecutionContext& Context) const;
};
