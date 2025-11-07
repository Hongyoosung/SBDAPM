// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Observation/ObservationElement.h"
#include "AI/MCTS/MCTSNode.h"
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
     * Calculate team-level reward for a given team observation
     * Considers team health, formation, objectives, combat effectiveness
     */
    float CalculateTeamReward(const FTeamObservation& TeamObs) const;

    /**
     * Generate strategic commands for followers based on team observation
     * Uses simple rule-based heuristics (placeholder for full MCTS implementation)
     */
    TMap<AActor*, struct FStrategicCommand> GenerateStrategicCommands(
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
};
