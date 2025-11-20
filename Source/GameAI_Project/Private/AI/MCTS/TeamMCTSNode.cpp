// Copyright Epic Games, Inc. All Rights Reserved.

#include "AI/MCTS/TeamMCTSNode.h"
#include "Kismet/KismetMathLibrary.h"

UTeamMCTSNode::UTeamMCTSNode()
	: Parent(nullptr)
	, TotalReward(0.0f)
	, VisitCount(0)
	, Depth(0)
{
}

void UTeamMCTSNode::Initialize(UTeamMCTSNode* InParent, const TMap<AActor*, FStrategicCommand>& InCommands)
{
	Parent = InParent;
	Commands = InCommands;
	TotalReward = 0.0f;
	VisitCount = 0;
	Depth = InParent ? InParent->Depth + 1 : 0;
	Children.Empty();
	UntriedActions.Empty();
}

bool UTeamMCTSNode::IsFullyExpanded() const
{
	return UntriedActions.Num() == 0;
}

bool UTeamMCTSNode::IsTerminal() const
{
	// Terminal if reached maximum depth
	const int32 MaxDepth = 5;
	return Depth >= MaxDepth;
}

UTeamMCTSNode* UTeamMCTSNode::SelectBestChild(float ExplorationParam) const
{
	if (Children.Num() == 0)
	{
		return nullptr;
	}

	UTeamMCTSNode* BestChild = nullptr;
	float BestValue = -FLT_MAX;

	for (UTeamMCTSNode* Child : Children)
	{
		if (!Child) continue;

		float UCTValue = Child->CalculateUCTValue(ExplorationParam);

		if (UCTValue > BestValue)
		{
			BestValue = UCTValue;
			BestChild = Child;
		}
	}

	return BestChild;
}

float UTeamMCTSNode::CalculateUCTValue(float ExplorationParam) const
{
	if (VisitCount == 0)
	{
		return FLT_MAX; // Unvisited nodes have infinite value
	}

	if (!Parent || Parent->VisitCount == 0)
	{
		return TotalReward / VisitCount; // Root node or invalid parent
	}

	// UCT formula: Q/N + C * sqrt(ln(N_parent) / N)
	float Exploitation = TotalReward / VisitCount;
	float Exploration = ExplorationParam * FMath::Sqrt(FMath::Loge(static_cast<float>(Parent->VisitCount)) / VisitCount);

	return Exploitation + Exploration;
}

UTeamMCTSNode* UTeamMCTSNode::Expand(const TArray<AActor*>& Followers)
{
	if (UntriedActions.Num() == 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("TeamMCTSNode::Expand: No untried actions available"));
		return nullptr;
	}

	// Pick a random untried action
	int32 RandomIndex = FMath::RandRange(0, UntriedActions.Num() - 1);
	TMap<AActor*, FStrategicCommand> NewCommands = UntriedActions[RandomIndex];
	UntriedActions.RemoveAt(RandomIndex);

	// Create child node
	UTeamMCTSNode* Child = NewObject<UTeamMCTSNode>(this);
	Child->Initialize(this, NewCommands);

	Children.Add(Child);

	return Child;
}

void UTeamMCTSNode::Backpropagate(float Reward)
{
	VisitCount++;
	TotalReward += Reward;

	if (Parent)
	{
		Parent->Backpropagate(Reward);
	}
}
