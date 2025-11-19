// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeTaskBase.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "STTask_Dead.generated.h"

class UAnimMontage;

/**
 * State Tree Task: Dead
 *
 * Executes death behavior when the follower agent dies.
 * - Plays death animation montage
 * - Destroys the agent after a configurable delay
 *
 * This task is entered when:
 * - STCondition_IsAlive returns false (bCheckIfDead = true)
 * - Health reaches 0
 *
 * The task completes by destroying the actor, so it never
 * returns Succeeded - the actor is simply removed.
 */


/**
 * Instance data for Dead task
 */
USTRUCT()
struct GAMEAI_PROJECT_API FSTTask_DeadInstanceData
{
	GENERATED_BODY()

	//--------------------------------------------------------------------------
	// CONTEXT BINDING (UE 5.6 - auto-binds to FollowerContext from schema)
	//--------------------------------------------------------------------------

	/**
	 * Shared context struct - automatically bound by StateTree
	 */
	UPROPERTY(EditAnywhere, Category = "Context")
	FFollowerStateTreeContext Context;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Death animation montage to play */
	UPROPERTY(EditAnywhere, Category = "Parameter")
	TObjectPtr<UAnimMontage> DeathMontage = nullptr;

	/** Time to wait before destroying the actor (seconds) */
	UPROPERTY(EditAnywhere, Category = "Parameter", meta = (ClampMin = "0.0", ClampMax = "30.0"))
	float DestroyDelay = 2.0f;

	/** Enable ragdoll physics on death */
	UPROPERTY(EditAnywhere, Category = "Parameter")
	bool bEnableRagdoll = false;

	/** Disable collision on death */
	UPROPERTY(EditAnywhere, Category = "Parameter")
	bool bDisableCollision = true;

	//--------------------------------------------------------------------------
	// RUNTIME STATE (managed by task)
	//--------------------------------------------------------------------------

	/** Time since death state entered */
	UPROPERTY()
	float TimeSinceDeath = 0.0f;

	/** Has the death animation started */
	UPROPERTY()
	bool bAnimationStarted = false;

	/** Has the actor been marked for destruction */
	UPROPERTY()
	bool bMarkedForDestruction = false;
};

USTRUCT(meta = (DisplayName = "Dead"))
struct GAMEAI_PROJECT_API FSTTask_Dead : public FStateTreeTaskBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTTask_DeadInstanceData;

	FSTTask_Dead() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }

	virtual EStateTreeRunStatus EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const override;
	virtual EStateTreeRunStatus Tick(FStateTreeExecutionContext& Context, float DeltaTime) const override;
	virtual void ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const override;

protected:
	/** Play the death animation montage */
	void PlayDeathAnimation(FStateTreeExecutionContext& Context) const;

	/** Enable ragdoll physics */
	void EnableRagdoll(FStateTreeExecutionContext& Context) const;

	/** Destroy the actor */
	void DestroyActor(FStateTreeExecutionContext& Context) const;
};
