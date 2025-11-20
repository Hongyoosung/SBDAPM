// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Observation/ObservationElement.h"
#include "AI/MCTS/MCTSNode.h"
#include "AI/MCTS/TeamMCTSNode.h"
#include "MCTS.generated.h"

struct FTeamObservation;

UCLASS()
class GAMEAI_PROJECT_API UMCTS : public UObject
{
	GENERATED_BODY()

public:
    UMCTS();

    //--------------------------------------------------------------------------
    // LEGACY SINGLE-AGENT INTERFACE (Backward Compatibility)
    //--------------------------------------------------------------------------
    void InitializeMCTS();
    void InitializeCurrentNodeLocate();
    void RunMCTS();
    void Backpropagate();


    //--------------------------------------------------------------------------
    // TEAM-LEVEL INTERFACE (New Architecture)
    //--------------------------------------------------------------------------
    /**
     * Initialize MCTS for team-level strategic decision making
     * @param InMaxSimulations - Number of MCTS simulations to run
     * @param InExplorationParam - UCT exploration parameter (default: 1.41)
     */
    void InitializeTeamMCTS(int32 InMaxSimulations = 500, float InExplorationParam = 1.41f);


    /**
     * Run team-level MCTS and return strategic commands for each follower
     * @param TeamObservation - Current team observation
     * @param Followers - List of follower actors
     * @return Map of follower to strategic command
     */
    TMap<AActor*, struct FStrategicCommand> RunTeamMCTS(
        const FTeamObservation& TeamObservation,
        const TArray<AActor*>& Followers
    );


private:
    //--------------------------------------------------------------------------
    // SINGLE-AGENT METHODS (Legacy)
    //--------------------------------------------------------------------------
    UMCTSNode* SelectChildNode();
    void Expand();
    float CalculateImmediateReward(UMCTSNode* Node) const;
    bool ShouldTerminate() const;
    float CalculateNodeScore(UMCTSNode* Node) const;
    float CalculateObservationSimilarity(const FObservationElement&, const FObservationElement&) const;
    float CalculateDynamicExplorationParameter() const;
    FObservationElement GetCurrentObservation();

    //--------------------------------------------------------------------------
    // TEAM-LEVEL METHODS (New)
    //--------------------------------------------------------------------------

    /**
     * Run full MCTS tree search with selection, expansion, simulation, backpropagation
     * @param TeamObs - Current team observation
     * @param Followers - List of followers
     * @return Best command assignment found
     */
    TMap<AActor*, struct FStrategicCommand> RunTeamMCTSTreeSearch(
        const FTeamObservation& TeamObs,
        const TArray<AActor*>& Followers
    );

    /**
     * MCTS Selection Phase: Traverse tree using UCT until leaf node
     */
    UTeamMCTSNode* SelectNode(UTeamMCTSNode* Node);

    /**
     * MCTS Expansion Phase: Create child node with untried action
     */
    UTeamMCTSNode* ExpandNode(UTeamMCTSNode* Node, const TArray<AActor*>& Followers);

    /**
     * MCTS Simulation Phase: Rollout from node to estimate reward
     */
    float SimulateNode(UTeamMCTSNode* Node, const FTeamObservation& TeamObs);

    /**
     * Generate possible command combinations for expansion
     * Uses smart sampling to avoid exponential explosion (11^N combinations)
     */
    TArray<TMap<AActor*, FStrategicCommand>> GenerateCommandCombinations(
        const TArray<AActor*>& Followers,
        const FTeamObservation& TeamObs,
        int32 MaxCombinations = 10
    ) const;

    /**
     * Calculate base team-level reward (no command-specific bonuses)
     * Considers team health, formation, objectives, combat effectiveness
     */
    float CalculateTeamReward(const FTeamObservation& TeamObs) const;

    /**
     * Calculate team-level reward for a given command assignment (with synergy bonuses)
     * Considers team health, formation, objectives, combat effectiveness, and command synergies
     */
    float CalculateTeamReward(const FTeamObservation& TeamObs, const TMap<AActor*, FStrategicCommand>& Commands) const;

    /**
     * FALLBACK: Generate strategic commands using rule-based heuristics
     * Used when MCTS tree search is disabled or fails
     */
    TMap<AActor*, struct FStrategicCommand> GenerateStrategicCommandsHeuristic(
        const FTeamObservation& TeamObs,
        const TArray<AActor*>& Followers
    ) const;



public:
    //--------------------------------------------------------------------------
    // CONFIGURATION (Team-Level)
    //--------------------------------------------------------------------------
    /** Maximum number of MCTS simulations to run */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MCTS|Config")
    int32 MaxSimulations;

    /** Discount factor for future rewards (0-1) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MCTS|Config")
    float DiscountFactor;

    /** UCT exploration parameter (sqrt(2) ≈ 1.41 recommended) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MCTS|Config")
    float ExplorationParameter;

    /** Enable full MCTS tree search (if false, uses fast heuristics) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MCTS|Config")
    bool bUseTreeSearch;

    /** Maximum command combinations to generate per expansion */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MCTS|Config")
    int32 MaxCombinationsPerExpansion;


private:
    //--------------------------------------------------------------------------
    // SINGLE-AGENT STATE (Legacy)
    //--------------------------------------------------------------------------
    UPROPERTY()
    TObjectPtr<UMCTSNode> RootNode;

    UPROPERTY()
    TObjectPtr<UMCTSNode> CurrentNode;

    UPROPERTY()
    FObservationElement CurrentObservation;

    uint32 TreeDepth;

    //--------------------------------------------------------------------------
    // TEAM-LEVEL STATE (New)
    //--------------------------------------------------------------------------
    /** Root node of team MCTS tree */
    UPROPERTY()
    TObjectPtr<UTeamMCTSNode> TeamRootNode;

    /** Cached team observation for simulation */
    FTeamObservation CachedTeamObservation;
};
