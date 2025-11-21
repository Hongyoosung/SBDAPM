#include "AI/MCTS/MCTS.h"
#include "Kismet/KismetMathLibrary.h"
#include "Team/TeamTypes.h"

UMCTS::UMCTS()
    : MaxSimulations(500)
    , DiscountFactor(0.95f)
    , ExplorationParameter(1.41f)
    , MaxCombinationsPerExpansion(10)
    , TeamRootNode(nullptr)
{
}

//==============================================================================
// TEAM-LEVEL MCTS IMPLEMENTATION
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
    UE_LOG(LogTemp, Log, TEXT("MCTS: Running team-level tree search for %d followers (%d simulations)"),
        Followers.Num(), MaxSimulations);

    // Always use full MCTS tree search
    return RunTeamMCTSTreeSearch(TeamObservation, Followers);
}


float UMCTS::CalculateTeamReward(const FTeamObservation& TeamObs) const
{
    // Team health component (0-200 points)
    float HealthReward = TeamObs.AverageTeamHealth * 2.0f;

    // Formation coherence (0-50 points)
    float FormationReward = TeamObs.FormationCoherence * 50.0f;

    // ============================================================================
    // PROXIMITY DIAGNOSIS: Log FormationCoherence value
    // ============================================================================
    UE_LOG(LogTemp, Warning, TEXT("[MCTS] FormationCoherence=%.3f (reward component: %.1f)"),
        TeamObs.FormationCoherence, FormationReward);

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
    // ============================================================================
    // RESEARCH BASELINE: Rule-Based Heuristic Decision Making
    // ============================================================================
    // This method is NOT part of MCTS tree search. It's a traditional rule-based
    // decision-making system used as a baseline for performance comparison.
    //
    // Use this to compare:
    // - MCTS exploration vs deterministic rules
    // - Learning-based strategies vs hand-crafted heuristics
    // ============================================================================

    TMap<AActor*, FStrategicCommand> Commands;

    UE_LOG(LogTemp, Warning, TEXT("📊 BASELINE HEURISTIC: Generating commands for %d followers"), Followers.Num());
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
    TeamRootNode = MakeShared<FTeamMCTSNode>();
    TMap<AActor*, FStrategicCommand> InitialCommands;
    TeamRootNode->Initialize(nullptr, InitialCommands);

    // Generate initial untried actions for root
    TeamRootNode->UntriedActions = GenerateCommandCombinations(Followers, TeamObs, MaxCombinationsPerExpansion);

    UE_LOG(LogTemp, Display, TEXT("🌲 MCTS: Root initialized with %d possible combinations"),
        TeamRootNode->UntriedActions.Num());

    // Early exit if no combinations generated
    if (TeamRootNode->UntriedActions.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("🌲 MCTS: No command combinations generated! Falling back to baseline heuristic"));
        return GenerateStrategicCommandsHeuristic(TeamObs, Followers);
    }

    // Run MCTS simulations
    for (int32 i = 0; i < MaxSimulations; ++i)
    {
        TSharedPtr<FTeamMCTSNode> LeafNode = SelectNode(TeamRootNode);

        if (!LeafNode.IsValid())
        {
            break;
        }

        TSharedPtr<FTeamMCTSNode> NodeToSimulate = LeafNode;

        if (!LeafNode->IsTerminal() && LeafNode->VisitCount > 0)
        {
            NodeToSimulate = ExpandNode(LeafNode, Followers);
            if (!NodeToSimulate.IsValid())
            {
                NodeToSimulate = LeafNode;
            }
        }

        float Reward = SimulateNode(NodeToSimulate, TeamObs);
        NodeToSimulate->Backpropagate(Reward);
    }

    TSharedPtr<FTeamMCTSNode> BestChild = nullptr;
    int32 MaxVisits = 0;

    for (const TSharedPtr<FTeamMCTSNode>& Child : TeamRootNode->Children)
    {
        if (Child.IsValid() && Child->VisitCount > MaxVisits)
        {
            MaxVisits = Child->VisitCount;
            BestChild = Child;
        }
    }

    if (BestChild.IsValid())
    {
        float AvgReward = BestChild->TotalReward / FMath::Max(1, BestChild->VisitCount);
        UE_LOG(LogTemp, Warning, TEXT("🌲 MCTS TREE SEARCH: Best child found (Visits: %d, Avg Reward: %.1f)"),
            MaxVisits, AvgReward);
        return BestChild->GetCommands();
    }
    else if (TeamRootNode->VisitCount > 0)
    {
        // Root was simulated but no children - return root commands
        UE_LOG(LogTemp, Warning, TEXT("🌲 MCTS: No children expanded, using root node commands"));
        return TeamRootNode->GetCommands();
    }
    else
    {
        // Complete failure - use baseline heuristic as emergency fallback
        UE_LOG(LogTemp, Error, TEXT("🌲 MCTS TREE SEARCH FAILED: No simulations completed! Using baseline heuristic fallback"));
        return GenerateStrategicCommandsHeuristic(TeamObs, Followers);
    }
}


TSharedPtr<FTeamMCTSNode> UMCTS::SelectNode(TSharedPtr<FTeamMCTSNode> Node)
{
    // Traverse tree using UCT until reaching a leaf node
    if (!Node.IsValid()) return nullptr;

    while (!Node->IsTerminal())
    {
        if (!Node->IsFullyExpanded())
        {
            return Node;
        }
        else
        {
            Node = Node->SelectBestChild(ExplorationParameter);
            if (!Node.IsValid())
            {
                break;
            }
        }
    }

    return Node;
}


TSharedPtr<FTeamMCTSNode> UMCTS::ExpandNode(TSharedPtr<FTeamMCTSNode> Node, const TArray<AActor*>& Followers)
{
    if (!Node.IsValid()) return nullptr;

    if (Node->UntriedActions.Num() == 0)
    {
        Node->UntriedActions = GenerateCommandCombinations(
            Followers,
            CachedTeamObservation,
            MaxCombinationsPerExpansion
        );
    }

    return Node->Expand(Followers);
}


float UMCTS::SimulateNode(TSharedPtr<FTeamMCTSNode> Node, const FTeamObservation& TeamObs)
{
    if (!Node.IsValid()) return 0.0f;

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

    if (Followers.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("GenerateCommandCombinations: No followers provided"));
        return Combinations;
    }

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
