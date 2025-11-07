// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Components/StateTreeComponent.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "FollowerStateTreeComponent.generated.h"

class UFollowerAgentComponent;

/**
 * Follower State Tree Component
 *
 * Manages State Tree execution for follower agents.
 * Replaces both StateMachine + BehaviorTree with a unified State Tree system.
 *
 * Architecture:
 * - Team Leader issues strategic commands
 * - FollowerAgentComponent receives commands
 * - FollowerStateTreeComponent executes commands via State Tree
 * - RL policy selects tactical actions within states
 *
 * Setup:
 * 1. Add this component to your AI pawn/character
 * 2. Assign StateTreeAsset (create in editor: Right Click > Gameplay > State Tree)
 * 3. Set FollowerComponent reference (auto-found if on same actor)
 * 4. Component auto-starts State Tree on BeginPlay
 *
 * State Tree Asset Structure:
 * Root (Selector)
 * ├─ [IsAlive == false] DeadState
 * ├─ [CommandType == Assault] AssaultState
 * │  ├─ QueryRLPolicy
 * │  └─ ExecuteAssault
 * ├─ [CommandType == Defend] DefendState
 * │  ├─ QueryRLPolicy
 * │  └─ ExecuteDefend
 * ├─ [CommandType == Support] SupportState
 * │  ├─ QueryRLPolicy
 * │  └─ ExecuteSupport
 * └─ IdleState
 *
 * Evaluators (Global):
 * - UpdateObservation (runs every tick, gathers 71-feature observation)
 * - SyncCommand (syncs command from FollowerAgentComponent)
 */
UCLASS(ClassGroup = "StateTree", meta = (BlueprintSpawnableComponent, DisplayName = "Follower StateTree Component"))
class GAMEAI_PROJECT_API UFollowerStateTreeComponent : public UStateTreeComponent
{
	GENERATED_BODY()

public:
	UFollowerStateTreeComponent();

	virtual void BeginPlay() override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	// Provide context data to State Tree execution
	virtual bool SetContextRequirements(FStateTreeExecutionContext& Context, bool bLogErrors = false) override;

	UFUNCTION(BlueprintPure, Category = "State Tree")
    virtual TSubclassOf<UStateTreeSchema> GetRequiredStateTreeSchema() const;


	//--------------------------------------------------------------------------
	// API
	//--------------------------------------------------------------------------

	/** Initialize State Tree context */
	UFUNCTION(BlueprintCallable, Category = "State Tree")
	void InitializeContext();

	/** Get State Tree context */
	UFUNCTION(BlueprintPure, Category = "State Tree")
	FFollowerStateTreeContext GetContext() const { return Context; }

	/** Update context from follower component */
	UFUNCTION(BlueprintCallable, Category = "State Tree")
	void UpdateContextFromFollower();

	/** Is State Tree currently running? */
	UFUNCTION(BlueprintPure, Category = "State Tree")
	bool IsStateTreeRunning() const;

	/** Get current state name (for debugging) */
	UFUNCTION(BlueprintPure, Category = "State Tree")
	FString GetCurrentStateName() const;

	/** Get current tactical action (from context) */
	UFUNCTION(BlueprintPure, Category = "State Tree")
	ETacticalAction GetCurrentTacticalAction() const { return Context.CurrentTacticalAction; }

	/** Get current command (from context) */
	UFUNCTION(BlueprintPure, Category = "State Tree")
	FStrategicCommand GetCurrentCommand() const { return Context.CurrentCommand; }

	// External data collection delegate
	bool CollectExternalData(
		const FStateTreeExecutionContext& InContext,
		const UStateTree* StateTree,
		TArrayView<const FStateTreeExternalDataDesc> ExternalDataDescs,
		TArrayView<FStateTreeDataView> OutDataViews
	);

protected:
	/** Find FollowerAgentComponent on actor */
	UFollowerAgentComponent* FindFollowerComponent();

	/** Bind to follower component events */
	void BindToFollowerEvents();

	/** Handle command received from leader */
	UFUNCTION()
	void OnCommandReceived(const FStrategicCommand& Command, EFollowerState NewState);

	/** Handle follower death */
	void OnFollowerDied();


public:
	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Follower agent component reference (auto-found if nullptr) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "State Tree")
	UFollowerAgentComponent* FollowerComponent = nullptr;

	/** Auto-find FollowerAgentComponent on same actor */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "State Tree")
	bool bAutoFindFollowerComponent = true;

	/** Auto-start State Tree on BeginPlay */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "State Tree")
	bool bAutoStartStateTree = true;

	//--------------------------------------------------------------------------
	// STATE TREE CONTEXT
	//--------------------------------------------------------------------------

	/** Shared context for State Tree (accessed by all tasks/evaluators/conditions) */
	UPROPERTY(BlueprintReadOnly, Category = "State Tree")
	FFollowerStateTreeContext Context;
};
