// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Team/TeamTypes.h"
#include "Observation/TeamObservation.h"
#include "TeamMCTSNode.generated.h"

/**
 * Team-level MCTS Node for strategic command search
 *
 * Represents a state in the team command space where each follower
 * has been assigned a strategic command. The MCTS algorithm searches
 * this space to find optimal command combinations.
 */
UCLASS()
class GAMEAI_PROJECT_API UTeamMCTSNode : public UObject
{
	GENERATED_BODY()

public:
	UTeamMCTSNode();

	/** Initialize node with parent and command assignment */
	void Initialize(UTeamMCTSNode* InParent, const TMap<AActor*, FStrategicCommand>& InCommands);

	/** Check if this node is fully expanded (all actions tried) */
	bool IsFullyExpanded() const;

	/** Check if this is a terminal node (simulation depth limit reached) */
	bool IsTerminal() const;

	/** Select best child using UCT formula */
	UTeamMCTSNode* SelectBestChild(float ExplorationParam) const;

	/** Expand this node by adding a new child with untried action */
	UTeamMCTSNode* Expand(const TArray<AActor*>& Followers);

	/** Backpropagate reward from simulation */
	void Backpropagate(float Reward);

	/** Get the command assignment for this node */
	TMap<AActor*, FStrategicCommand> GetCommands() const { return Commands; }

	/** Calculate UCT value for this node */
	float CalculateUCTValue(float ExplorationParam) const;

public:
	/** Parent node (nullptr for root) */
	UPROPERTY()
	TObjectPtr<UTeamMCTSNode> Parent;

	/** Child nodes (expanded actions) */
	UPROPERTY()
	TArray<TObjectPtr<UTeamMCTSNode>> Children;

	/** Command assignment for this node (follower -> command) */
	TMap<AActor*, FStrategicCommand> Commands;

	/** Total reward accumulated from simulations */
	float TotalReward;

	/** Number of times this node has been visited */
	int32 VisitCount;

	/** Depth of this node in the tree (0 = root) */
	int32 Depth;

	/** List of untried command combinations */
	TArray<TMap<AActor*, FStrategicCommand>> UntriedActions;
};
