#include "AI/MCTS.h"
#include "Kismet/KismetMathLibrary.h"
#include "States/FleeState.h"
#include "States/AttackState.h"
#include "States/MoveToState.h"
#include "Core/StateMachine.h"
#include "Observation/TeamObservation.h"
#include "Team/TeamTypes.h"

UMCTS::UMCTS()
    : RootNode(nullptr), CurrentNode(nullptr), TreeDepth(0)
    , MaxSimulations(500), DiscountFactor(0.95f), ExplorationParameter(1.41f)
{
}


void UMCTS::InitializeMCTS()
{
    RootNode = NewObject<UMCTSNode>(this);
    RootNode->InitializeNode(nullptr, nullptr);

    FPlatformProcess::Sleep(0.2f);

    if (RootNode != nullptr)
    {
        RootNode->VisitCount = 1;

        UE_LOG(LogTemp, Warning, TEXT("Initialized MCTS"));
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("Failed to Initialize MCTS"));
    }
}


void UMCTS::InitializeCurrentNodeLocate()
{
    UE_LOG(LogTemp, Warning, TEXT("Initialize Current Node Locate"));
    CurrentNode = RootNode;
    TreeDepth = 1;
}


UMCTSNode* UMCTS::SelectChildNode()
{
    if(ShouldTerminate())
	{
		UE_LOG(LogTemp, Warning, TEXT("ShouldTerminate is true, cannot select child node"));
		return nullptr;
	}


    UMCTSNode* BestChild = nullptr;
    float BestScore = -FLT_MAX;


    if (CurrentNode->Children.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("CurrentNode has no children"));
        return nullptr;
    }


    for (UMCTSNode* Child : CurrentNode->Children)
    {

        if (Child == nullptr)
        {
            UE_LOG(LogTemp, Error, TEXT("Child node is nullptr"));
            continue;
        }

        float Score = CalculateNodeScore(Child);

        if (Score > BestScore)
        {
            BestScore = Score;
            BestChild = Child;
        }
    }

    if (BestChild)
    {
        UE_LOG(LogTemp, Warning, TEXT("Selected Child with UCT Value: %f"), BestScore);

        return BestChild;
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("No valid child found"));
        return nullptr;
    }
}



float UMCTS::CalculateNodeScore(UMCTSNode* Node) const
{
    if(Node == nullptr)
    {
        UE_LOG(LogTemp, Warning,
        TEXT("Node is nullptr, cannot calculate node score"));
        return -FLT_MAX;
    }

    if (Node->VisitCount == 0)
        return FLT_MAX;  // �湮���� ���� ��� �켱 Ž��

    if(Node->Parent == nullptr)
	{
		UE_LOG(LogTemp, Warning, TEXT("Parent node is nullptr, cannot calculate node score"));
		return -FLT_MAX;
	}


    float Exploitation = Node->TotalReward / Node->VisitCount;
    float Exploration = ExplorationParameter * FMath::Sqrt(FMath::Loge((double)Node->Parent->VisitCount) / Node->VisitCount);
    float ObservationSimilarity = CalculateObservationSimilarity(Node->Observation, CurrentObservation);
    float Recency = 1.0f / (1.0f + Node->LastVisitTime);  // �ֱ� �湮�ϼ��� ���� ��

    // ���� Ž�� �Ķ���� ���
    float DynamicExplorationParameter = CalculateDynamicExplorationParameter();


    return Exploitation + DynamicExplorationParameter * Exploration * ObservationSimilarity ;
}


float UMCTS::CalculateDynamicExplorationParameter() const
{

    float DepthFactor = FMath::Max(0.5f, 1.0f - (TreeDepth / 20.0f));


    float AverageReward = (RootNode->TotalReward / RootNode->VisitCount);
    float RewardFactor = FMath::Max(0.5f, 1.0f - (AverageReward / 100.0f));  // 100�� �ִ� ���� ����


    //float TimeFactor = FMath::Max(0.5f, 1.0f - (World->GetWorld()->GetTimeSeconds() / 300.0f));  // 300�� �� �ּҰ�

    return ExplorationParameter * DepthFactor * RewardFactor;
}


float UMCTS::CalculateObservationSimilarity(const FObservationElement& Obs1, const FObservationElement& Obs2) const
{
    // �� ����� ���̸� ����ϰ� ����ȭ
    float DistanceDiff = FMath::Abs(Obs1.DistanceToDestination - Obs2.DistanceToDestination) / 100.0f; // �Ÿ��� 100���� ������ ����ȭ
    float HealthDiff = FMath::Abs(Obs1.AgentHealth - Obs2.AgentHealth) / 100.0f; // ü���� 100���� ������ ����ȭ
    float EnemiesDiff = FMath::Abs(Obs1.VisibleEnemyCount - Obs2.VisibleEnemyCount) / 10.0f; // ���� ���� 10���� ������ ����ȭ

    // �� ��ҿ� ����ġ ����
    const float DistanceWeight = 0.4f;
    const float HealthWeight = 0.4f;
    const float EnemiesWeight = 0.2f;

    // ���� ����ư �Ÿ� ���
    float WeightedDistance = DistanceWeight * DistanceDiff + HealthWeight * HealthDiff + EnemiesWeight * EnemiesDiff;

    // �Ÿ��� ���絵�� ��ȯ (���� ���� �Լ� ���)
    return FMath::Exp(-WeightedDistance * 5.0f);  // 5.0f�� ���� �ӵ��� �����ϴ� �Ķ����
}


void UMCTS::Expand(TArray<UAction*> PossibleActions)
{
    for (UAction* PossibleAction : PossibleActions)
    {
        UMCTSNode* NewNode = NewObject<UMCTSNode>(this);
        if (NewNode != nullptr)
        {
            NewNode->InitializeNode(CurrentNode, PossibleAction);
            NewNode->Observation = CurrentObservation; // �׼ǿ� ���� ������ ������ ����
            CurrentNode->Children.Add(NewNode);
            UE_LOG(LogTemp, Warning, TEXT("Expand: Create New Node"));
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("Failed to create a new node"));
        }
    }
}


void UMCTS::Backpropagate()
{
    int Depth = 0;

    while (CurrentNode != RootNode)
    {
        CurrentNode->VisitCount++;

        // ȿ���ġ ���: ��� ����� ���ε� �̷� ������ ���� ���
        float ImmediateReward = CalculateImmediateReward(CurrentNode);
        float DiscountedFutureReward = ImmediateReward * FMath::Pow(DiscountFactor, Depth);
        float WeightedReward = (ImmediateReward + DiscountedFutureReward) / 2.0f;

        CurrentNode->TotalReward += WeightedReward;
        UE_LOG(LogTemp, Warning, TEXT("Backpropagate: Update Node - VisitCount: %d, TotalReward: %f"),
			CurrentNode->VisitCount, CurrentNode->TotalReward);

        CurrentNode = CurrentNode->Parent;
        Depth++;
    }

    TreeDepth = 1;
}


float UMCTS::CalculateImmediateReward(UMCTSNode* Node) const
{
    // Note: We need to access StateMachine to determine current state
    // For now, we'll use a simpler approach based on observation data
    // TODO: Pass StateMachine reference to this function for state-specific rewards

    const FObservationElement& Obs = Node->Observation;

    // Default reward calculation (for MoveToState, AttackState, etc.)
    float BaseDistanceReward = 100.0f - Obs.DistanceToDestination;
    float BaseHealthReward = Obs.AgentHealth;
    float BaseEnemyPenalty = -10.0f * Obs.VisibleEnemyCount;

    // Detect if this is likely a flee scenario based on observation characteristics
    // Heuristic: If health is low (<40) AND enemies are numerous (>2), likely fleeing
    bool bLikelyFleeScenario = (Obs.AgentHealth < 40.0f) && (Obs.VisibleEnemyCount > 2);

    if (bLikelyFleeScenario)
    {
        // FLEE-SPECIFIC REWARD CALCULATION
        // When fleeing, prioritize survival and distance from enemies

        // 1. Cover Availability Reward
        float CoverReward = Obs.bHasCover ? 100.0f : 0.0f;

        // 2. Distance from Enemies Reward
        // Calculate average enemy distance from NearbyEnemies array
        float TotalEnemyDistance = 0.0f;
        int32 ValidEnemies = 0;
        for (const FEnemyObservation& Enemy : Obs.NearbyEnemies)
        {
            if (Enemy.Distance < 3000.0f) // Only count nearby enemies (within 30m)
            {
                TotalEnemyDistance += Enemy.Distance;
                ValidEnemies++;
            }
        }
        float AvgEnemyDistance = ValidEnemies > 0 ? TotalEnemyDistance / ValidEnemies : 3000.0f;
        // Normalize distance reward (farther = better)
        float DistanceFromEnemiesReward = AvgEnemyDistance / 50.0f; // Scale to 0-60 range

        // 3. Health Preservation Reward
        // Reward maintaining health (penalize damage taken)
        // Note: We'd need previous health to calculate this properly
        // For now, just reward higher health during flee
        float HealthPreservationReward = Obs.AgentHealth * 0.5f; // Higher health = better

        // 4. Stamina Penalty (can't sprint effectively if low stamina)
        float StaminaPenalty = (Obs.Stamina < 20.0f) ? -30.0f : 0.0f;

        // 5. Cover Distance Reward (prefer closer cover when available)
        float CoverDistanceReward = 0.0f;
        if (Obs.bHasCover)
        {
            // Closer cover is better (max reward at 0 distance, min at 1500cm)
            CoverDistanceReward = FMath::Max(0.0f, 50.0f - (Obs.NearestCoverDistance / 30.0f));
        }

        float FleeReward = CoverReward + DistanceFromEnemiesReward +
                          HealthPreservationReward + StaminaPenalty + CoverDistanceReward;

        UE_LOG(LogTemp, Verbose, TEXT("MCTS Flee Reward: Total=%.1f (Cover=%.1f, DistFromEnemy=%.1f, Health=%.1f, Stamina=%.1f, CoverDist=%.1f)"),
            FleeReward, CoverReward, DistanceFromEnemiesReward, HealthPreservationReward, StaminaPenalty, CoverDistanceReward);

        return FleeReward;
    }
    else
    {
        // DEFAULT REWARD CALCULATION (Attack, MoveTo, etc.)
        float DefaultReward = BaseDistanceReward + BaseHealthReward + BaseEnemyPenalty;

        UE_LOG(LogTemp, Verbose, TEXT("MCTS Default Reward: Total=%.1f (Distance=%.1f, Health=%.1f, Enemy=%.1f)"),
            DefaultReward, BaseDistanceReward, BaseHealthReward, BaseEnemyPenalty);

        return DefaultReward;
    }
}


// ���� ���� ���� �Լ�
bool UMCTS::ShouldTerminate() const
{
    // currentnode�� null�̸� �ߴ�
    if (CurrentNode == nullptr)
	{
        UE_LOG(LogTemp, Warning, TEXT("ShouldTerminate: CurrentNode is nullptr"));
		return true;
	}


    // Ʈ�� ���̰� 10�� �Ѿ�� �ߴ�
    if (TreeDepth >= 10)
	{
        UE_LOG(LogTemp, Warning, TEXT("ShouldTerminate: TreeDepth is over 10"));

		return true;
	}

    // �� ���� ������ �߰��� �� ����
    return false;
}


void UMCTS::RunMCTS(TArray<UAction*> PossibleActions, UStateMachine* StateMachine)
{
    UE_LOG(LogTemp, Warning, TEXT("RunMCTS Start - CurrentNode: %p, TreeDepth: %d"), CurrentNode, TreeDepth);

    // Ʈ�� ���� ���ѿ� �����ߴ��� Ȯ��
    if (ShouldTerminate())
    {
        // ������ ����
        Backpropagate();

        UE_LOG(LogTemp, Warning, TEXT("Tree depth limit reached. Returning to root node."));
    }

    // Ȯ�� �ܰ�
    if (CurrentNode != nullptr && CurrentNode->Children.IsEmpty() && !ShouldTerminate())
    {
        Expand(PossibleActions);
    }

    FPlatformProcess::Sleep(0.2f);

    UMCTSNode* BestChild = SelectChildNode();
    TreeDepth++;

    FPlatformProcess::Sleep(0.2f);

    UE_LOG(LogTemp, Warning, TEXT("Before ExecuteAction - BestChild: %p, Action: %p"), BestChild, BestChild ? BestChild->Action : nullptr);

    if (CurrentNode && BestChild && BestChild->Action)
    {
        CurrentNode = BestChild;

        FPlatformProcess::Sleep(0.2f);

        CurrentNode->Action->ExecuteAction(StateMachine);
        UE_LOG(LogTemp, Warning, TEXT("After ExecuteAction - CurrentNode: %p, TreeDepth: %d"), CurrentNode, TreeDepth);
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to execute action - CurrentNode: %p, BestChild: %p, Action: %p"),
            CurrentNode, BestChild, BestChild ? BestChild->Action : nullptr);
    }
}


FObservationElement UMCTS::GetCurrentObservation(UStateMachine* StateMachine)
{
    // ���� ���� ���¿� ���� ������ ����
    FObservationElement Observation{};

    Observation.DistanceToDestination = StateMachine->DistanceToDestination;
    Observation.AgentHealth = StateMachine->AgentHealth;
    Observation.VisibleEnemyCount = StateMachine->EnemiesNum;

    return Observation;
}


//==============================================================================
// TEAM-LEVEL MCTS IMPLEMENTATION (New Architecture)
//==============================================================================

void UMCTS::InitializeTeamMCTS(int32 InMaxSimulations, float InExplorationParam)
{
    MaxSimulations = InMaxSimulations;
    ExplorationParameter = InExplorationParam;

    UE_LOG(LogTemp, Log, TEXT("MCTS: Initialized for team-level decisions (Simulations: %d, Exploration: %.2f)"),
        MaxSimulations, ExplorationParameter);
}


TMap<AActor*, FStrategicCommand> UMCTS::RunTeamMCTS(
    const FTeamObservation& TeamObservation,
    const TArray<AActor*>& Followers)
{
    UE_LOG(LogTemp, Log, TEXT("MCTS: Running team-level search for %d followers"), Followers.Num());

    // TODO: Implement full MCTS tree search for team-level action space
    // Current implementation uses rule-based heuristics as placeholder
    //
    // Full implementation would:
    // 1. Create root node with current team observation
    // 2. For each simulation:
    //    a. Selection: UCT-based node selection
    //    b. Expansion: Generate child nodes for command combinations
    //    c. Simulation: Evaluate team reward
    //    d. Backpropagation: Update node statistics
    // 3. Select best command combination based on visit counts/rewards

    return GenerateStrategicCommands(TeamObservation, Followers);
}


float UMCTS::CalculateTeamReward(const FTeamObservation& TeamObs) const
{
    // Team health component (0-200 points)
    float HealthReward = TeamObs.AverageTeamHealth * 2.0f;

    // Formation coherence (0-50 points)
    float FormationReward = TeamObs.FormationCoherence * 50.0f;

    // Objective progress (0-100 points)
    // Closer to objective = higher reward
    float ObjectiveReward = 0.0f;
    if (TeamObs.DistanceToObjective > 0.0f)
    {
        // Max reward at 0 distance, min at 10000cm (100m)
        ObjectiveReward = FMath::Max(0.0f, 100.0f - (TeamObs.DistanceToObjective / 100.0f));
    }

    // Combat effectiveness (variable, can be negative)
    float CombatReward = TeamObs.KillDeathRatio * 50.0f;

    // Threat penalty (-100 to 0 points)
    float ThreatPenalty = -TeamObs.ThreatLevel * 20.0f;

    // Outnumbered penalty (-100 points)
    float OutnumberedPenalty = TeamObs.bOutnumbered ? -100.0f : 0.0f;

    // Flanked penalty (-50 points)
    float FlankedPenalty = TeamObs.bFlanked ? -50.0f : 0.0f;

    // Cover advantage bonus (+50 points)
    float CoverBonus = TeamObs.bHasCoverAdvantage ? 50.0f : 0.0f;

    // High ground bonus (+30 points)
    float HighGroundBonus = TeamObs.bHasHighGround ? 30.0f : 0.0f;

    float TotalReward = HealthReward + FormationReward + ObjectiveReward + CombatReward
                      + ThreatPenalty + OutnumberedPenalty + FlankedPenalty
                      + CoverBonus + HighGroundBonus;

    UE_LOG(LogTemp, Verbose, TEXT("MCTS Team Reward: %.1f (Health=%.1f, Formation=%.1f, Objective=%.1f, Combat=%.1f, Threat=%.1f)"),
        TotalReward, HealthReward, FormationReward, ObjectiveReward, CombatReward, ThreatPenalty);

    return TotalReward;
}


TMap<AActor*, FStrategicCommand> UMCTS::GenerateStrategicCommands(
    const FTeamObservation& TeamObs,
    const TArray<AActor*>& Followers) const
{
    TMap<AActor*, FStrategicCommand> Commands;

    UE_LOG(LogTemp, Warning, TEXT("📊 MCTS: Generating commands for %d followers"), Followers.Num());
    UE_LOG(LogTemp, Warning, TEXT("  Team State: Enemies=%d, Health=%.1f%%, Outnumbered=%s, Flanked=%s, DistToObj=%.1f"),
        TeamObs.TotalVisibleEnemies,
        TeamObs.AverageTeamHealth,
        TeamObs.bOutnumbered ? TEXT("Yes") : TEXT("No"),
        TeamObs.bFlanked ? TEXT("Yes") : TEXT("No"),
        TeamObs.DistanceToObjective);

    // Rule-based command generation (placeholder for full MCTS)
    // This implements simple tactical AI based on team situation

    for (AActor* Follower : Followers)
    {
        if (!Follower) continue;

        FStrategicCommand Command;

        // Decision logic based on team observation
        if (TeamObs.TotalVisibleEnemies > 0)
        {
            // COMBAT SITUATION
            if (TeamObs.bOutnumbered && TeamObs.AverageTeamHealth < 60.0f)
            {
                // Outnumbered and low health - retreat
                Command.CommandType = EStrategicCommandType::Retreat;
                Command.Priority = 9;
                UE_LOG(LogTemp, Verbose, TEXT("MCTS: Follower %s - RETREAT (outnumbered, low health)"), *Follower->GetName());
            }
            else if (TeamObs.bFlanked)
            {
                // Being flanked - regroup
                Command.CommandType = EStrategicCommandType::Regroup;
                Command.TargetLocation = TeamObs.TeamCentroid;
                Command.Priority = 8;
                UE_LOG(LogTemp, Verbose, TEXT("MCTS: Follower %s - REGROUP (flanked)"), *Follower->GetName());
            }
            else if (TeamObs.AverageTeamHealth > 70.0f && !TeamObs.bOutnumbered)
            {
                // Good health, not outnumbered - assault
                Command.CommandType = EStrategicCommandType::Assault;
                Command.Priority = 7;
                UE_LOG(LogTemp, Verbose, TEXT("MCTS: Follower %s - ASSAULT (healthy, advantage)"), *Follower->GetName());
            }
            else
            {
                // Default combat - take cover and suppress
                Command.CommandType = EStrategicCommandType::TakeCover;
                Command.Priority = 6;
                UE_LOG(LogTemp, Verbose, TEXT("MCTS: Follower %s - TAKE COVER (neutral combat)"), *Follower->GetName());
            }
        }
        else if (TeamObs.AverageTeamHealth < 50.0f)
        {
            // NO ENEMIES, LOW HEALTH - hold position and recover
            Command.CommandType = EStrategicCommandType::HoldPosition;
            Command.Priority = 5;
            UE_LOG(LogTemp, Verbose, TEXT("MCTS: Follower %s - HOLD POSITION (recovering)"), *Follower->GetName());
        }
        else if (TeamObs.DistanceToObjective > 1000.0f)
        {
            // FAR FROM OBJECTIVE - advance
            Command.CommandType = EStrategicCommandType::Advance;
            Command.Priority = 5;
            UE_LOG(LogTemp, Verbose, TEXT("MCTS: Follower %s - ADVANCE (toward objective)"), *Follower->GetName());
        }
        else if (TeamObs.FormationCoherence < 0.5f)
        {
            // FORMATION BROKEN - regroup
            Command.CommandType = EStrategicCommandType::Regroup;
            Command.TargetLocation = TeamObs.TeamCentroid;
            Command.Priority = 4;
            UE_LOG(LogTemp, Verbose, TEXT("MCTS: Follower %s - REGROUP (formation broken)"), *Follower->GetName());
        }
        else
        {
            // DEFAULT - patrol
            Command.CommandType = EStrategicCommandType::Patrol;
            Command.Priority = 3;
            UE_LOG(LogTemp, Verbose, TEXT("MCTS: Follower %s - PATROL (default)"), *Follower->GetName());
        }

        Commands.Add(Follower, Command);
    }

    return Commands;
}
