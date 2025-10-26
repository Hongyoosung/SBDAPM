// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "AIController.h"
#include "BehaviorTree/BehaviorTree.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "BehaviorTree/BlackboardData.h"
#include "SBDAPMController.generated.h"

class UStateMachine;
class UBehaviorTree;

/**
 * ASBDAPMController - AI Controller for SBDAPM agents
 *
 * This controller bridges the strategic decision-making layer (FSM + MCTS)
 * with the tactical execution layer (Behavior Tree). It manages:
 * - Starting and running the Behavior Tree
 * - Providing access to the Blackboard for state communication
 * - Connecting to the StateMachine component on the controlled pawn
 *
 * Architecture:
 * - Strategic Layer (FSM + MCTS): Makes high-level decisions (attack, flee, move)
 * - Communication Layer (Blackboard): Bridges strategic and tactical layers
 * - Tactical Layer (Behavior Tree): Executes low-level actions (pathfinding, aiming, shooting)
 */
UCLASS()
class GAMEAI_PROJECT_API ASBDAPMController : public AAIController
{
	GENERATED_BODY()

public:
	ASBDAPMController();

protected:
	virtual void BeginPlay() override;
	virtual void OnPossess(APawn* InPawn) override;
	virtual void OnUnPossess() override;

public:
	/**
	 * The Behavior Tree asset to run for this AI agent.
	 * This should be set in Blueprint or via C++ before the controller possesses a pawn.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI|Behavior Tree")
	TObjectPtr<UBehaviorTree> BehaviorTreeAsset;

	/**
	 * The Blackboard asset to use with the Behavior Tree.
	 * This should match the Blackboard referenced by the BehaviorTreeAsset.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI|Behavior Tree")
	TObjectPtr<UBlackboardData> BlackboardAsset;

	/**
	 * Whether to automatically start the Behavior Tree when possessing a pawn.
	 * Default: true
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI|Behavior Tree")
	bool bAutoStartBehaviorTree = true;

	/**
	 * Get the StateMachine component from the controlled pawn.
	 * Returns nullptr if the pawn doesn't have a StateMachine component.
	 */
	UFUNCTION(BlueprintCallable, Category = "AI|SBDAPM")
	UStateMachine* GetStateMachine() const;

	/**
	 * Manually start the Behavior Tree.
	 * Called automatically in OnPossess if bAutoStartBehaviorTree is true.
	 */
	UFUNCTION(BlueprintCallable, Category = "AI|Behavior Tree")
	bool StartBehaviorTree();

	/**
	 * Stop the currently running Behavior Tree.
	 */
	UFUNCTION(BlueprintCallable, Category = "AI|Behavior Tree")
	void StopBehaviorTree();

	/**
	 * Check if the Behavior Tree is currently running.
	 */
	UFUNCTION(BlueprintPure, Category = "AI|Behavior Tree")
	bool IsBehaviorTreeRunning() const;

	/**
	 * Set a Blackboard key value (String).
	 * Convenience method for strategic layer to communicate with Behavior Tree.
	 */
	UFUNCTION(BlueprintCallable, Category = "AI|Blackboard")
	void SetBlackboardValueAsString(FName KeyName, const FString& Value);

	/**
	 * Set a Blackboard key value (Vector).
	 */
	UFUNCTION(BlueprintCallable, Category = "AI|Blackboard")
	void SetBlackboardValueAsVector(FName KeyName, FVector Value);

	/**
	 * Set a Blackboard key value (Object).
	 */
	UFUNCTION(BlueprintCallable, Category = "AI|Blackboard")
	void SetBlackboardValueAsObject(FName KeyName, UObject* Value);

	/**
	 * Set a Blackboard key value (Float).
	 */
	UFUNCTION(BlueprintCallable, Category = "AI|Blackboard")
	void SetBlackboardValueAsFloat(FName KeyName, float Value);

	/**
	 * Set a Blackboard key value (Bool).
	 */
	UFUNCTION(BlueprintCallable, Category = "AI|Blackboard")
	void SetBlackboardValueAsBool(FName KeyName, bool Value);

	/**
	 * Get a Blackboard key value (String).
	 */
	UFUNCTION(BlueprintPure, Category = "AI|Blackboard")
	FString GetBlackboardValueAsString(FName KeyName) const;

	/**
	 * Get a Blackboard key value (Vector).
	 */
	UFUNCTION(BlueprintPure, Category = "AI|Blackboard")
	FVector GetBlackboardValueAsVector(FName KeyName) const;

	/**
	 * Get a Blackboard key value (Object).
	 */
	UFUNCTION(BlueprintPure, Category = "AI|Blackboard")
	UObject* GetBlackboardValueAsObject(FName KeyName) const;

	/**
	 * Get a Blackboard key value (Float).
	 */
	UFUNCTION(BlueprintPure, Category = "AI|Blackboard")
	float GetBlackboardValueAsFloat(FName KeyName) const;

	/**
	 * Get a Blackboard key value (Bool).
	 */
	UFUNCTION(BlueprintPure, Category = "AI|Blackboard")
	bool GetBlackboardValueAsBool(FName KeyName) const;

private:
	/** Cached reference to the StateMachine component on the controlled pawn */
	UPROPERTY()
	UStateMachine* CachedStateMachine;
};
