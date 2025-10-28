// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Observation/ObservationElement.h"
#include "RLTypes.generated.h"

/**
 * Tactical actions that can be selected by the RL policy
 * These are lower-level actions executed within a strategic context
 */
UENUM(BlueprintType, meta=(UseAsBlackboardKey))
enum class ETacticalAction : uint8
{
	// Combat Tactics (used in Assault state)
	AggressiveAssault UMETA(DisplayName = "Aggressive Assault"),
	CautiousAdvance UMETA(DisplayName = "Cautious Advance"),
	DefensiveHold UMETA(DisplayName = "Defensive Hold"),
	TacticalRetreat UMETA(DisplayName = "Tactical Retreat"),

	// Positioning Tactics
	SeekCover UMETA(DisplayName = "Seek Cover"),
	FlankLeft UMETA(DisplayName = "Flank Left"),
	FlankRight UMETA(DisplayName = "Flank Right"),
	MaintainDistance UMETA(DisplayName = "Maintain Distance"),

	// Support Tactics
	SuppressiveFire UMETA(DisplayName = "Suppressive Fire"),
	ProvideCoveringFire UMETA(DisplayName = "Provide Covering Fire"),
	Reload UMETA(DisplayName = "Reload"),
	UseAbility UMETA(DisplayName = "Use Ability"),

	// Movement Tactics
	Sprint UMETA(DisplayName = "Sprint"),
	Crouch UMETA(DisplayName = "Crouch"),
	Patrol UMETA(DisplayName = "Patrol"),
	Hold UMETA(DisplayName = "Hold")
};

/**
 * Experience tuple for reinforcement learning
 * Represents a single transition in the MDP
 */
USTRUCT(BlueprintType)
struct FRLExperience
{
	GENERATED_BODY()

	// Current state (71 features)
	UPROPERTY(BlueprintReadWrite, Category = "RL")
	FObservationElement State;

	// Action taken in this state
	UPROPERTY(BlueprintReadWrite, Category = "RL")
	ETacticalAction Action;

	// Immediate reward received
	UPROPERTY(BlueprintReadWrite, Category = "RL")
	float Reward;

	// Next state after taking action (71 features)
	UPROPERTY(BlueprintReadWrite, Category = "RL")
	FObservationElement NextState;

	// Is this a terminal state?
	UPROPERTY(BlueprintReadWrite, Category = "RL")
	bool bTerminal;

	// Timestamp of experience
	UPROPERTY(BlueprintReadWrite, Category = "RL")
	float Timestamp;

	// Additional context data (optional)
	UPROPERTY(BlueprintReadWrite, Category = "RL")
	TMap<FString, float> ContextData;

	FRLExperience()
		: Action(ETacticalAction::DefensiveHold)
		, Reward(0.0f)
		, bTerminal(false)
		, Timestamp(0.0f)
	{
	}

	FRLExperience(const FObservationElement& InState, ETacticalAction InAction, float InReward, const FObservationElement& InNextState, bool bInTerminal)
		: State(InState)
		, Action(InAction)
		, Reward(InReward)
		, NextState(InNextState)
		, bTerminal(bInTerminal)
		, Timestamp(0.0f)
	{
	}
};

/**
 * RL training statistics
 */
USTRUCT(BlueprintType)
struct FRLTrainingStats
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite, Category = "RL")
	int32 TotalExperiences;

	UPROPERTY(BlueprintReadWrite, Category = "RL")
	int32 EpisodesCompleted;

	UPROPERTY(BlueprintReadWrite, Category = "RL")
	float AverageReward;

	UPROPERTY(BlueprintReadWrite, Category = "RL")
	float AverageEpisodeLength;

	UPROPERTY(BlueprintReadWrite, Category = "RL")
	float LastEpisodeReward;

	UPROPERTY(BlueprintReadWrite, Category = "RL")
	float BestEpisodeReward;

	UPROPERTY(BlueprintReadWrite, Category = "RL")
	float TrainingTimeSeconds;

	FRLTrainingStats()
		: TotalExperiences(0)
		, EpisodesCompleted(0)
		, AverageReward(0.0f)
		, AverageEpisodeLength(0.0f)
		, LastEpisodeReward(0.0f)
		, BestEpisodeReward(-MAX_FLT)
		, TrainingTimeSeconds(0.0f)
	{
	}
};

/**
 * RL policy configuration
 */
USTRUCT(BlueprintType)
struct FRLPolicyConfig
{
	GENERATED_BODY()

	// Number of input features (should be 71)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL")
	int32 InputSize;

	// Number of output actions (should be 16)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL")
	int32 OutputSize;

	// Hidden layer sizes
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL")
	TArray<int32> HiddenLayers;

	// Learning rate for training
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL")
	float LearningRate;

	// Discount factor (gamma)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL")
	float DiscountFactor;

	// Epsilon for epsilon-greedy exploration
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL")
	float Epsilon;

	// Epsilon decay rate
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL")
	float EpsilonDecay;

	// Minimum epsilon value
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL")
	float MinEpsilon;

	// Path to ONNX model file
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL")
	FString ModelPath;

	FRLPolicyConfig()
		: InputSize(71)
		, OutputSize(16)
		, HiddenLayers({128, 128, 64})
		, LearningRate(0.0003f)
		, DiscountFactor(0.99f)
		, Epsilon(1.0f)
		, EpsilonDecay(0.995f)
		, MinEpsilon(0.05f)
		, ModelPath(TEXT(""))
	{
	}
};

/**
 * Reward components for tactical actions
 */
USTRUCT(BlueprintType)
struct FTacticalRewards
{
	GENERATED_BODY()

	// Combat rewards
	static constexpr float KILL_ENEMY = 10.0f;
	static constexpr float DAMAGE_ENEMY = 5.0f;
	static constexpr float SUPPRESS_ENEMY = 3.0f;
	static constexpr float TAKE_DAMAGE = -5.0f;
	static constexpr float DIE = -10.0f;

	// Tactical rewards
	static constexpr float REACH_COVER = 5.0f;
	static constexpr float MAINTAIN_FORMATION = 3.0f;
	static constexpr float FOLLOW_COMMAND = 2.0f;
	static constexpr float BREAK_FORMATION = -3.0f;
	static constexpr float IGNORE_COMMAND = -5.0f;

	// Support rewards
	static constexpr float RESCUE_ALLY = 10.0f;
	static constexpr float COVERING_FIRE = 5.0f;
	static constexpr float SHARE_AMMO = 3.0f;

	// Efficiency penalties
	static constexpr float WASTED_AMMO = -1.0f;
	static constexpr float OUT_OF_POSITION = -2.0f;
	static constexpr float IDLE_TOO_LONG = -1.0f;
};
