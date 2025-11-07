#include "AI/MCTS/MCTSNode.h"

UMCTSNode::UMCTSNode()
    : Parent(nullptr), VisitCount(0), TotalReward(0.0f)
{
}

void UMCTSNode::InitializeNode(UMCTSNode* InParent)
{
    Parent = InParent;
    VisitCount = 0;
    TotalReward = 0.0f;
    Children.Empty();
}

FString UMCTSNode::GetState() const
{

    return TEXT("Root Node");

}
