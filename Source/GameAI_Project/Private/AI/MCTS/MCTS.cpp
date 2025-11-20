#include "AI/MCTS/MCTS.h"
#include "Kismet/KismetMathLibrary.h"
#include "Observation/TeamObservation.h"
#include "Team/TeamTypes.h"

UMCTS::UMCTS()
    : MaxSimulations(500)
    , DiscountFactor(0.95f)
    , ExplorationParameter(1.41f)
    , bUseTreeSearch(false)
    , MaxCombinationsPerExpansion(10)
    , RootNode(nullptr)
    , CurrentNode(nullptr)
    , TreeDepth(0)
    , TeamRootNode(nullptr)
{
}


void UMCTS::InitializeMCTS()
{
    RootNode = NewObject<UMCTSNode>(this);
    RootNode->InitializeNode(nullptr);

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
        return FLT_MAX; 

    if(Node->Parent == nullptr)
	{
		UE_LOG(LogTemp, Warning, TEXT("Parent node is nullptr, cannot calculate node score"));
		return -FLT_MAX;
	}


    float Exploitation = Node->TotalReward / Node->VisitCount;
    float Exploration = ExplorationParameter * FMath::Sqrt(FMath::Loge((double)Node->Parent->VisitCount) / Node->VisitCount);
    float ObservationSimilarity = CalculateObservationSimilarity(Node->Observation, CurrentObservation);
    float Recency = 1.0f / (1.0f + Node->LastVisitTime); 

    // ���� Ž�� �Ķ���� ���
    float DynamicExplorationParameter = CalculateDynamicExplorationParameter();


    return Exploitation + DynamicExplorationParameter * Exploration * ObservationSimilarity ;
}


float UMCTS::CalculateDynamicExplorationParameter() const
{

    float DepthFactor = FMath::Max(0.5f, 1.0f - (TreeDepth / 20.0f));


    float AverageReward = (RootNode->TotalReward / RootNode->VisitCount);
    float RewardFactor = FMath::Max(0.5f, 1.0f - (AverageReward / 100.0f));  


    return ExplorationParameter * DepthFactor * RewardFactor;
}


float UMCTS::CalculateObservationSimilarity(const FObservationElement& Obs1, const FObservationElement& Obs2) const
{

    float DistanceDiff = FMath::Abs(Obs1.DistanceToDestination - Obs2.DistanceToDestination) / 100.0f;
    float HealthDiff = FMath::Abs(Obs1.AgentHealth - Obs2.AgentHealth) / 100.0f; 
    float EnemiesDiff = FMath::Abs(Obs1.VisibleEnemyCount - Obs2.VisibleEnemyCount) / 10.0f; 


    const float DistanceWeight = 0.4f;
    const float HealthWeight = 0.4f;
    const float EnemiesWeight = 0.2f;


    float WeightedDistance = DistanceWeight * DistanceDiff + HealthWeight * HealthDiff + EnemiesWeight * EnemiesDiff;


    return FMath::Exp(-WeightedDistance * 5.0f);  
}


void UMCTS::Expand()
{

}


void UMCTS::Backpropagate()
{
    int Depth = 0;

    while (CurrentNode != RootNode)
    {
        CurrentNode->VisitCount++;

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


bool UMCTS::ShouldTerminate() const
{
    
    if (CurrentNode == nullptr)
	{
        UE_LOG(LogTemp, Warning, TEXT("ShouldTerminate: CurrentNode is nullptr"));
		return true;
	}


    if (TreeDepth >= 10)
	{
        UE_LOG(LogTemp, Warning, TEXT("ShouldTerminate: TreeDepth is over 10"));

		return true;
	}

    return false;
}


void UMCTS::RunMCTS()
{
    UE_LOG(LogTemp, Warning, TEXT("RunMCTS Start - CurrentNode: %p, TreeDepth: %d"), CurrentNode.Get(), TreeDepth);


    if (ShouldTerminate())
    {
        Backpropagate();

        UE_LOG(LogTemp, Warning, TEXT("Tree depth limit reached. Returning to root node."));
    }


    FPlatformProcess::Sleep(0.2f);

    UMCTSNode* BestChild = SelectChildNode();
    TreeDepth++;

    FPlatformProcess::Sleep(0.2f);

}


FObservationElement UMCTS::GetCurrentObservation()
{

	return FObservationElement();
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
    UE_LOG(LogTemp, Log, TEXT("MCTS: Running team-level search for %d followers (TreeSearch: %s)"),
        Followers.Num(), bUseTreeSearch ? TEXT("ENABLED") : TEXT("DISABLED"));

    if (bUseTreeSearch)
    {
        // Use full MCTS tree search
        return RunTeamMCTSTreeSearch(TeamObservation, Followers);
    }
    else
    {
        // Use fast heuristic-based command generation
        return GenerateStrategicCommandsHeuristic(TeamObservation, Followers);
    }
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


TMap<AActor*, FStrategicCommand> UMCTS::GenerateStrategicCommandsHeuristic(
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

    // Rule-based command generation with tactical diversity
    // Assign different roles to followers based on team composition

    // Shuffle followers to randomize role assignment
    TArray<AActor*> ShuffledFollowers = Followers;
    for (int32 i = ShuffledFollowers.Num() - 1; i > 0; --i)
    {
        int32 j = FMath::RandRange(0, i);
        ShuffledFollowers.Swap(i, j);
    }

    int32 FollowerIndex = 0;
    int32 NumFollowers = ShuffledFollowers.Num();

    for (AActor* Follower : ShuffledFollowers)
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

                // Assign nearest enemy as target (to retreat away from and suppress)
                AActor* NearestEnemy = nullptr;
                float NearestDistance = FLT_MAX;
                FVector FollowerLocation = Follower->GetActorLocation();

                for (AActor* Enemy : TeamObs.TrackedEnemies)
                {
                    if (Enemy && Enemy->IsValidLowLevel())
                    {
                        float Distance = FVector::Dist(FollowerLocation, Enemy->GetActorLocation());
                        if (Distance < NearestDistance)
                        {
                            NearestDistance = Distance;
                            NearestEnemy = Enemy;
                        }
                    }
                }

                if (NearestEnemy)
                {
                    Command.TargetActor = NearestEnemy;
                    UE_LOG(LogTemp, Warning, TEXT("MCTS: Follower %s - RETREAT - Target: %s (Distance: %.1f)"),
                        *Follower->GetName(), *NearestEnemy->GetName(), NearestDistance);
                }
                else
                {
                    UE_LOG(LogTemp, Warning, TEXT("MCTS: Follower %s - RETREAT (no valid target found)"), *Follower->GetName());
                }
            }
            else if (TeamObs.bFlanked)
            {
                // Being flanked - regroup
                Command.CommandType = EStrategicCommandType::Regroup;
                Command.TargetLocation = TeamObs.TeamCentroid;
                Command.Priority = 8;
                UE_LOG(LogTemp, Warning, TEXT("MCTS: Follower %s - REGROUP (flanked)"), *Follower->GetName());
            }
            else if (TeamObs.AverageTeamHealth > 70.0f && !TeamObs.bOutnumbered)
            {
                // Good health, not outnumbered - diversify tactics
                // Calculate role distribution based on team size and enemy count
                float RoleRatio = static_cast<float>(FollowerIndex) / FMath::Max(1, NumFollowers - 1);

                // Role assignment:
                // - First ~50% assault (aggressive engagement)
                // - Next ~30% support (suppression/assistance)
                // - Last ~20% defend (cover/overwatch)
                int32 NumAssault = FMath::CeilToInt(NumFollowers * 0.5f);
                int32 NumSupport = FMath::CeilToInt(NumFollowers * 0.3f);

                if (FollowerIndex < NumAssault)
                {
                    // ASSAULT role - engage enemies directly
                    Command.CommandType = EStrategicCommandType::Assault;
                    Command.Priority = 7;

                    // Assign nearest enemy as target
                    AActor* NearestEnemy = nullptr;
                    float NearestDistance = FLT_MAX;
                    FVector FollowerLocation = Follower->GetActorLocation();

                    for (AActor* Enemy : TeamObs.TrackedEnemies)
                    {
                        if (Enemy && Enemy->IsValidLowLevel())
                        {
                            float Distance = FVector::Dist(FollowerLocation, Enemy->GetActorLocation());
                            if (Distance < NearestDistance)
                            {
                                NearestDistance = Distance;
                                NearestEnemy = Enemy;
                            }
                        }
                    }

                    if (NearestEnemy)
                    {
                        Command.TargetActor = NearestEnemy;
                        Command.TargetLocation = NearestEnemy->GetActorLocation();
                        UE_LOG(LogTemp, Warning, TEXT("MCTS: Follower %s [%d/%d] - ASSAULT - Target: %s (Distance: %.1f)"),
                            *Follower->GetName(), FollowerIndex + 1, NumFollowers, *NearestEnemy->GetName(), NearestDistance);
                    }
                    else
                    {
                        UE_LOG(LogTemp, Warning, TEXT("MCTS: Follower %s [%d/%d] - ASSAULT (no valid target found)"),
                            *Follower->GetName(), FollowerIndex + 1, NumFollowers);
                    }
                }
                else if (FollowerIndex < NumAssault + NumSupport)
                {
                    // SUPPORT role - provide fire support
                    Command.CommandType = EStrategicCommandType::Support;
                    Command.Priority = 6;
                    Command.TargetLocation = TeamObs.TeamCentroid;

                    // Assign nearest enemy as target for fire support
                    AActor* NearestEnemy = nullptr;
                    float NearestDistance = FLT_MAX;
                    FVector FollowerLocation = Follower->GetActorLocation();

                    for (AActor* Enemy : TeamObs.TrackedEnemies)
                    {
                        if (Enemy && Enemy->IsValidLowLevel())
                        {
                            float Distance = FVector::Dist(FollowerLocation, Enemy->GetActorLocation());
                            if (Distance < NearestDistance)
                            {
                                NearestDistance = Distance;
                                NearestEnemy = Enemy;
                            }
                        }
                    }

                    if (NearestEnemy)
                    {
                        Command.TargetActor = NearestEnemy;
                        UE_LOG(LogTemp, Warning, TEXT("MCTS: Follower %s [%d/%d] - SUPPORT - Target: %s (Distance: %.1f)"),
                            *Follower->GetName(), FollowerIndex + 1, NumFollowers, *NearestEnemy->GetName(), NearestDistance);
                    }
                    else
                    {
                        UE_LOG(LogTemp, Warning, TEXT("MCTS: Follower %s [%d/%d] - SUPPORT (no valid target found)"),
                            *Follower->GetName(), FollowerIndex + 1, NumFollowers);
                    }
                }
                else
                {
                    // DEFEND role - take cover and provide overwatch
                    Command.CommandType = EStrategicCommandType::TakeCover;
                    Command.Priority = 6;

                    // Assign nearest enemy as target for overwatch
                    AActor* NearestEnemy = nullptr;
                    float NearestDistance = FLT_MAX;
                    FVector FollowerLocation = Follower->GetActorLocation();

                    for (AActor* Enemy : TeamObs.TrackedEnemies)
                    {
                        if (Enemy && Enemy->IsValidLowLevel())
                        {
                            float Distance = FVector::Dist(FollowerLocation, Enemy->GetActorLocation());
                            if (Distance < NearestDistance)
                            {
                                NearestDistance = Distance;
                                NearestEnemy = Enemy;
                            }
                        }
                    }

                    if (NearestEnemy)
                    {
                        Command.TargetActor = NearestEnemy;
                        UE_LOG(LogTemp, Warning, TEXT("MCTS: Follower %s [%d/%d] - TAKE COVER - Target: %s (Distance: %.1f)"),
                            *Follower->GetName(), FollowerIndex + 1, NumFollowers, *NearestEnemy->GetName(), NearestDistance);
                    }
                    else
                    {
                        UE_LOG(LogTemp, Warning, TEXT("MCTS: Follower %s [%d/%d] - TAKE COVER (no valid target found)"),
                            *Follower->GetName(), FollowerIndex + 1, NumFollowers);
                    }
                }
            }
            else
            {
                // Default combat - take cover and suppress
                Command.CommandType = EStrategicCommandType::TakeCover;
                Command.Priority = 6;

                // Assign nearest enemy as target
                AActor* NearestEnemy = nullptr;
                float NearestDistance = FLT_MAX;
                FVector FollowerLocation = Follower->GetActorLocation();

                for (AActor* Enemy : TeamObs.TrackedEnemies)
                {
                    if (Enemy && Enemy->IsValidLowLevel())
                    {
                        float Distance = FVector::Dist(FollowerLocation, Enemy->GetActorLocation());
                        if (Distance < NearestDistance)
                        {
                            NearestDistance = Distance;
                            NearestEnemy = Enemy;
                        }
                    }
                }

                if (NearestEnemy)
                {
                    Command.TargetActor = NearestEnemy;
                    UE_LOG(LogTemp, Warning, TEXT("MCTS: Follower %s - TAKE COVER - Target: %s (Distance: %.1f)"),
                        *Follower->GetName(), *NearestEnemy->GetName(), NearestDistance);
                }
                else
                {
                    UE_LOG(LogTemp, Warning, TEXT("MCTS: Follower %s - TAKE COVER (no valid target found)"), *Follower->GetName());
                }
            }
        }
        else if (TeamObs.AverageTeamHealth < 50.0f)
        {
            // NO ENEMIES, LOW HEALTH - hold position and recover
            Command.CommandType = EStrategicCommandType::HoldPosition;
            Command.Priority = 5;
            UE_LOG(LogTemp, Warning, TEXT("MCTS: Follower %s - HOLD POSITION (recovering)"), *Follower->GetName());
        }
        else if (TeamObs.DistanceToObjective > 1000.0f)
        {
            // FAR FROM OBJECTIVE - advance
            Command.CommandType = EStrategicCommandType::Advance;
            Command.Priority = 5;
            UE_LOG(LogTemp, Warning, TEXT("MCTS: Follower %s - ADVANCE (toward objective)"), *Follower->GetName());
        }
        else if (TeamObs.FormationCoherence < 0.5f)
        {
            // FORMATION BROKEN - regroup
            Command.CommandType = EStrategicCommandType::Regroup;
            Command.TargetLocation = TeamObs.TeamCentroid;
            Command.Priority = 4;
            UE_LOG(LogTemp, Warning, TEXT("MCTS: Follower %s - REGROUP (formation broken)"), *Follower->GetName());
        }
        else
        {
            // DEFAULT - patrol
            Command.CommandType = EStrategicCommandType::Patrol;
            Command.Priority = 3;
            UE_LOG(LogTemp, Warning, TEXT("MCTS: Follower %s - PATROL (default)"), *Follower->GetName());
        }

        Commands.Add(Follower, Command);
        FollowerIndex++;
    }

    return Commands;
}


//==============================================================================
// MCTS TREE SEARCH IMPLEMENTATION
//==============================================================================

TMap<AActor*, FStrategicCommand> UMCTS::RunTeamMCTSTreeSearch(
    const FTeamObservation& TeamObs,
    const TArray<AActor*>& Followers)
{
    UE_LOG(LogTemp, Warning, TEXT("🌲 MCTS TREE SEARCH: Starting for %d followers (%d simulations)"),
        Followers.Num(), MaxSimulations);

    // Cache observation for simulation phase
    CachedTeamObservation = TeamObs;

    // Create root node with empty command assignment
    TeamRootNode = NewObject<UTeamMCTSNode>(this);
    TMap<AActor*, FStrategicCommand> InitialCommands;
    TeamRootNode->Initialize(nullptr, InitialCommands);

    // Generate initial untried actions for root
    TeamRootNode->UntriedActions = GenerateCommandCombinations(Followers, TeamObs, MaxCombinationsPerExpansion);

    UE_LOG(LogTemp, Display, TEXT("🌲 MCTS: Root initialized with %d possible combinations"),
        TeamRootNode->UntriedActions.Num());

    // Run MCTS simulations
    for (int32 i = 0; i < MaxSimulations; ++i)
    {
        // 1. SELECTION: Traverse tree using UCT
        UTeamMCTSNode* LeafNode = SelectNode(TeamRootNode);

        // 2. EXPANSION: Add child node if not terminal
        UTeamMCTSNode* NodeToSimulate = LeafNode;
        if (!LeafNode->IsTerminal() && LeafNode->VisitCount > 0)
        {
            NodeToSimulate = ExpandNode(LeafNode, Followers);
            if (!NodeToSimulate)
            {
                NodeToSimulate = LeafNode; // Expansion failed, use leaf
            }
        }

        // 3. SIMULATION: Estimate reward
        float Reward = SimulateNode(NodeToSimulate, TeamObs);

        // 4. BACKPROPAGATION: Update node statistics
        NodeToSimulate->Backpropagate(Reward);

        if (i % 100 == 0)
        {
            UE_LOG(LogTemp, Verbose, TEXT("🌲 MCTS: Simulation %d/%d - Reward: %.1f"),
                i, MaxSimulations, Reward);
        }
    }

    // Select best child based on visit count (most explored = most promising)
    UTeamMCTSNode* BestChild = nullptr;
    int32 MaxVisits = 0;

    for (UTeamMCTSNode* Child : TeamRootNode->Children)
    {
        if (Child && Child->VisitCount > MaxVisits)
        {
            MaxVisits = Child->VisitCount;
            BestChild = Child;
        }
    }

    if (BestChild)
    {
        float AvgReward = BestChild->TotalReward / FMath::Max(1, BestChild->VisitCount);
        UE_LOG(LogTemp, Warning, TEXT("🌲 MCTS TREE SEARCH: Best child found (Visits: %d, Avg Reward: %.1f)"),
            MaxVisits, AvgReward);
        return BestChild->GetCommands();
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("🌲 MCTS TREE SEARCH: No children expanded! Falling back to heuristics"));
        return GenerateStrategicCommandsHeuristic(TeamObs, Followers);
    }
}


UTeamMCTSNode* UMCTS::SelectNode(UTeamMCTSNode* Node)
{
    // Traverse tree using UCT until reaching a leaf node
    while (!Node->IsTerminal())
    {
        if (!Node->IsFullyExpanded())
        {
            // Node has untried actions, return it for expansion
            return Node;
        }
        else
        {
            // Node is fully expanded, select best child using UCT
            Node = Node->SelectBestChild(ExplorationParameter);
            if (!Node)
            {
                UE_LOG(LogTemp, Error, TEXT("SelectNode: SelectBestChild returned nullptr!"));
                break;
            }
        }
    }

    return Node;
}


UTeamMCTSNode* UMCTS::ExpandNode(UTeamMCTSNode* Node, const TArray<AActor*>& Followers)
{
    if (Node->UntriedActions.Num() == 0)
    {
        // Generate new actions if none available
        Node->UntriedActions = GenerateCommandCombinations(
            Followers,
            CachedTeamObservation,
            MaxCombinationsPerExpansion
        );
    }

    return Node->Expand(Followers);
}


float UMCTS::SimulateNode(UTeamMCTSNode* Node, const FTeamObservation& TeamObs)
{
    // Fast rollout: evaluate reward for this command assignment
    TMap<AActor*, FStrategicCommand> Commands = Node->GetCommands();

    // If no commands assigned yet, use heuristic
    if (Commands.Num() == 0)
    {
        return CalculateTeamReward(TeamObs) * 0.5f; // Base reward
    }

    // Calculate reward based on command synergy and team state
    return CalculateTeamReward(TeamObs, Commands);
}


TArray<TMap<AActor*, FStrategicCommand>> UMCTS::GenerateCommandCombinations(
    const TArray<AActor*>& Followers,
    const FTeamObservation& TeamObs,
    int32 MaxCombinations) const
{
    TArray<TMap<AActor*, FStrategicCommand>> Combinations;

    // Available command types (strategic level)
    TArray<EStrategicCommandType> PossibleCommands = {
        EStrategicCommandType::Assault,
        EStrategicCommandType::TakeCover,
        EStrategicCommandType::Support,
        EStrategicCommandType::Retreat,
        EStrategicCommandType::Advance,
        EStrategicCommandType::HoldPosition
    };

    // Generate diverse command combinations using sampling
    for (int32 i = 0; i < MaxCombinations; ++i)
    {
        TMap<AActor*, FStrategicCommand> Combo;

        for (AActor* Follower : Followers)
        {
            if (!Follower) continue;

            // Pick random command type
            int32 RandomIndex = FMath::RandRange(0, PossibleCommands.Num() - 1);
            EStrategicCommandType CommandType = PossibleCommands[RandomIndex];

            FStrategicCommand Command;
            Command.CommandType = CommandType;
            Command.Priority = 7;

            // Assign target if enemies available
            if (TeamObs.TrackedEnemies.Num() > 0)
            {

                TArray<AActor*> TrackedEnemiesArray = TeamObs.TrackedEnemies.Array();

                int32 EnemyIndex = FMath::RandRange(0, TrackedEnemiesArray.Num() - 1);

                // 3. TArray에서 타겟을 가져옵니다.
                Command.TargetActor = TrackedEnemiesArray[EnemyIndex];

                if (Command.TargetActor)
                {
                    Command.TargetLocation = Command.TargetActor->GetActorLocation();
                }
            }

            Combo.Add(Follower, Command);
        }

        Combinations.Add(Combo);
    }

    UE_LOG(LogTemp, Verbose, TEXT("Generated %d command combinations for %d followers"),
        Combinations.Num(), Followers.Num());

    return Combinations;
}


float UMCTS::CalculateTeamReward(const FTeamObservation& TeamObs, const TMap<AActor*, FStrategicCommand>& Commands) const
{
    // Base team reward
    float BaseReward = CalculateTeamReward(TeamObs);

    // Command synergy bonuses
    float SynergyBonus = 0.0f;

    // Count command types
    TMap<EStrategicCommandType, int32> CommandCounts;
    for (const auto& Pair : Commands)
    {
        CommandCounts.FindOrAdd(Pair.Value.CommandType, 0)++;
    }

    int32 TotalFollowers = Commands.Num();

    // Bonus for tactical diversity (avoid all-assault or all-defend)
    if (CommandCounts.Num() >= 2)
    {
        SynergyBonus += 20.0f; // +20 for having multiple tactics
    }

    // Bonus for balanced composition in combat
    if (TeamObs.TotalVisibleEnemies > 0)
    {
        int32 AssaultCount = CommandCounts.FindRef(EStrategicCommandType::Assault);
        int32 SupportCount = CommandCounts.FindRef(EStrategicCommandType::Support);
        int32 DefendCount = CommandCounts.FindRef(EStrategicCommandType::TakeCover);

        // Reward having frontline + support + defense mix
        if (AssaultCount > 0 && (SupportCount > 0 || DefendCount > 0))
        {
            SynergyBonus += 30.0f; // +30 for combined arms
        }
    }

    // Penalty for poor situational choices
    if (TeamObs.bOutnumbered && CommandCounts.FindRef(EStrategicCommandType::Assault) == TotalFollowers)
    {
        SynergyBonus -= 50.0f; // -50 for all-assault when outnumbered
    }

    if (TeamObs.AverageTeamHealth < 40.0f && CommandCounts.FindRef(EStrategicCommandType::Retreat) == 0)
    {
        SynergyBonus -= 30.0f; // -30 for not retreating when low health
    }

    return BaseReward + SynergyBonus;
}
