// Copyright Epic Games, Inc. All Rights Reserved.

#include "AI/MCTS/TeamMCTSNode.h"
#include "Kismet/KismetMathLibrary.h"

FTeamMCTSNode::FTeamMCTSNode()
	: TotalReward(0.0f)
	, VisitCount(0)
	, Depth(0)
{
}

void FTeamMCTSNode::Initialize(TSharedPtr<FTeamMCTSNode> InParent, const TMap<AActor*, FStrategicCommand>& InCommands)
{
	Parent = InParent;
	Commands = InCommands;
	TotalReward = 0.0f;
	VisitCount = 0;

	TSharedPtr<FTeamMCTSNode> ParentPinned = InParent;
	Depth = ParentPinned.IsValid() ? ParentPinned->Depth + 1 : 0;

	Children.Empty();
	UntriedActions.Empty();
}

bool FTeamMCTSNode::IsFullyExpanded() const
{
	return UntriedActions.Num() == 0;
}

bool FTeamMCTSNode::IsTerminal() const
{
	// Terminal if reached maximum depth
	const int32 MaxDepth = 5;
	return Depth >= MaxDepth;
}

TSharedPtr<FTeamMCTSNode> FTeamMCTSNode::SelectBestChild(float ExplorationParam) const
{
	if (Children.Num() == 0)
	{
		return nullptr;
	}

	TSharedPtr<FTeamMCTSNode> BestChild = nullptr;
	float BestValue = -FLT_MAX;

	for (const TSharedPtr<FTeamMCTSNode>& Child : Children)
	{
		if (!Child.IsValid()) continue;

		float UCTValue = Child->CalculateUCTValue(ExplorationParam);

		if (UCTValue > BestValue)
		{
			BestValue = UCTValue;
			BestChild = Child;
		}
	}

	return BestChild;
}

float FTeamMCTSNode::CalculateUCTValue(float ExplorationParam) const
{
	if (VisitCount == 0) return FLT_MAX;

	TSharedPtr<FTeamMCTSNode> ParentPinned = Parent.Pin();
	if (!ParentPinned || ParentPinned->VisitCount == 0)
	{
		return TotalReward / VisitCount;
	}

	float Exploitation = TotalReward / VisitCount;
	float Exploration = ExplorationParam * FMath::Sqrt(FMath::Loge(static_cast<float>(ParentPinned->VisitCount)) / VisitCount);

	return Exploitation + Exploration;
}

TSharedPtr<FTeamMCTSNode> FTeamMCTSNode::Expand(const TArray<AActor*>& Followers)
{
	if (UntriedActions.Num() == 0) return nullptr;

	int32 RandomIndex = FMath::RandRange(0, UntriedActions.Num() - 1);
	TMap<AActor*, FStrategicCommand> NewCommands = UntriedActions[RandomIndex];
	UntriedActions.RemoveAt(RandomIndex);

	// 중요: NewObject 대신 MakeShared 사용
	TSharedPtr<FTeamMCTSNode> Child = MakeShared<FTeamMCTSNode>();

	// 'this'를 SharedPtr로 전달하기 위해 AsShared() 사용
	Child->Initialize(AsShared(), NewCommands);

	Children.Add(Child);

	return Child;
}

void FTeamMCTSNode::Backpropagate(float Reward)
{
	VisitCount++;
	TotalReward += Reward;

	TSharedPtr<FTeamMCTSNode> ParentPinned = Parent.Pin();
	if (ParentPinned.IsValid())
	{
		ParentPinned->Backpropagate(Reward);
	}
}
