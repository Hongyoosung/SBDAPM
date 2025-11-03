#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "TeamTypes.h"
#include "Observation/ObservationElement.h"
#include "RL/RLTypes.h"
#include "FollowerAgentComponent.generated.h"

// Forward declarations
class UTeamLeaderComponent;
class UStateMachine;
class UBehaviorTree;
class URLPolicyNetwork;

/**
 * Delegate for follower events
 */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(
	FOnCommandReceived,
	const FStrategicCommand&, Command,
	EFollowerState, NewState
);

DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(
	FOnEventSignaled,
	EStrategicEvent, Event,
	AActor*, Instigator,
	int32, Priority
);

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(
	FOnStateChanged,
	EFollowerState, OldState,
	EFollowerState, NewState
);

/**
 * Follower Agent Component - Tactical Execution
 *
 * Responsibilities:
 * - Receive strategic commands from team leader
 * - Transition FSM based on commands
 * - Maintain local observation (71 features)
 * - Signal strategic events to leader
 * - Execute commands via Behavior Tree
 *
 * Usage:
 * 1. Attach to an AI-controlled Actor (e.g., AGameAICharacter)
 * 2. Set TeamLeader reference
 * 3. Component will automatically register with leader
 * 4. Leader will issue commands, component will execute
 */
UCLASS(ClassGroup=(AI), meta=(BlueprintSpawnableComponent))
class GAMEAI_PROJECT_API UFollowerAgentComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UFollowerAgentComponent();

	virtual void BeginPlay() override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Team leader actor (will find TeamLeaderComponent on this actor) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Follower|Config")
	AActor* TeamLeaderActor = nullptr;

	/** Team leader component reference (auto-set from TeamLeaderActor) */
	UPROPERTY(BlueprintReadOnly, Category = "Follower|Config")
	UTeamLeaderComponent* TeamLeader = nullptr;

	/** Automatically register with team leader on BeginPlay */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Follower|Config")
	bool bAutoRegisterWithLeader = true;

	/** Auto-find team leader by tag (if TeamLeaderActor not set) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Follower|Config")
	FName TeamLeaderTag = TEXT("TeamLeader");

	/** State machine reference (existing FSM system) */
	UPROPERTY(BlueprintReadWrite, Category = "Follower|Components")
	UStateMachine* StrategicFSM = nullptr;

	/** Behavior tree to run (for command execution) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Follower|Components")
	UBehaviorTree* BehaviorTreeAsset = nullptr;

	/** RL policy for tactical action selection */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Follower|Components")
	URLPolicyNetwork* TacticalPolicy = nullptr;

	/** Enable RL policy (if false, uses rule-based fallback) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Follower|RL")
	bool bUseRLPolicy = true;

	/** Collect experiences for offline training */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Follower|RL")
	bool bCollectExperiences = true;

	/** Enable debug visualization */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Follower|Debug")
	bool bEnableDebugDrawing = false;

	//--------------------------------------------------------------------------
	// STATE
	//--------------------------------------------------------------------------

	/** Current follower state */
	UPROPERTY(BlueprintReadOnly, Category = "Follower|State")
	EFollowerState CurrentFollowerState = EFollowerState::Idle;

	/** Current command from leader */
	UPROPERTY(BlueprintReadOnly, Category = "Follower|State")
	FStrategicCommand CurrentCommand;

	/** Local observation (71 features) */
	UPROPERTY(BlueprintReadOnly, Category = "Follower|State")
	FObservationElement LocalObservation;

	/** Is follower alive? */
	UPROPERTY(BlueprintReadOnly, Category = "Follower|State")
	bool bIsAlive = true;

	/** Time since last command received */
	UPROPERTY(BlueprintReadOnly, Category = "Follower|State")
	float TimeSinceLastCommand = 0.0f;

	/** Last tactical action selected by RL policy */
	UPROPERTY(BlueprintReadOnly, Category = "Follower|RL")
	ETacticalAction LastTacticalAction = ETacticalAction::DefensiveHold;

	/** Time since last tactical action was taken (seconds) */
	UPROPERTY(BlueprintReadOnly, Category = "Follower|RL")
	float TimeSinceLastTacticalAction = 0.0f;

	/** Previous observation (for experience collection) */
	UPROPERTY(BlueprintReadOnly, Category = "Follower|RL")
	FObservationElement PreviousObservation;

	/** Accumulated reward this episode */
	UPROPERTY(BlueprintReadOnly, Category = "Follower|RL")
	float AccumulatedReward = 0.0f;

	//--------------------------------------------------------------------------
	// EVENTS
	//--------------------------------------------------------------------------

	/** Fired when command is received from leader */
	UPROPERTY(BlueprintAssignable, Category = "Follower|Events")
	FOnCommandReceived OnCommandReceived;

	/** Fired when event is signaled to leader */
	UPROPERTY(BlueprintAssignable, Category = "Follower|Events")
	FOnEventSignaled OnEventSignaled;

	/** Fired when follower state changes */
	UPROPERTY(BlueprintAssignable, Category = "Follower|Events")
	FOnStateChanged OnStateChanged;

	//--------------------------------------------------------------------------
	// TEAM LEADER COMMUNICATION
	//--------------------------------------------------------------------------

	/** Register with team leader */
	UFUNCTION(BlueprintCallable, Category = "Follower|Team")
	bool RegisterWithTeamLeader();

	/** Unregister from team leader */
	UFUNCTION(BlueprintCallable, Category = "Follower|Team")
	void UnregisterFromTeamLeader();

	/** Signal event to team leader */
	UFUNCTION(BlueprintCallable, Category = "Follower|Team")
	void SignalEventToLeader(
		EStrategicEvent Event,
		AActor* Instigator = nullptr,
		FVector Location = FVector::ZeroVector,
		int32 Priority = 5
	);

	/** Report command completion */
	UFUNCTION(BlueprintCallable, Category = "Follower|Team")
	void ReportCommandComplete(bool bSuccess = true);

	/** Request assistance from team */
	UFUNCTION(BlueprintCallable, Category = "Follower|Team")
	void RequestAssistance(int32 Priority = 8);

	//--------------------------------------------------------------------------
	// COMMAND EXECUTION
	//--------------------------------------------------------------------------

	/** Execute command received from leader */
	UFUNCTION(BlueprintCallable, Category = "Follower|Commands")
	void ExecuteCommand(const FStrategicCommand& Command);

	/** Get current command */
	UFUNCTION(BlueprintPure, Category = "Follower|Commands")
	FStrategicCommand GetCurrentCommand() const { return CurrentCommand; }

	/** Is command still valid? */
	UFUNCTION(BlueprintPure, Category = "Follower|Commands")
	bool IsCommandValid() const;

	/** Has active command (not Idle/None)? */
	UFUNCTION(BlueprintPure, Category = "Follower|Commands")
	bool HasActiveCommand() const;

	/** Update command progress */
	UFUNCTION(BlueprintCallable, Category = "Follower|Commands")
	void UpdateCommandProgress(float Progress);

	/** Get time since last command received (seconds) */
	UFUNCTION(BlueprintPure, Category = "Follower|Commands")
	float GetTimeSinceLastCommand() const { return TimeSinceLastCommand; }

	//--------------------------------------------------------------------------
	// STATE MANAGEMENT
	//--------------------------------------------------------------------------

	/** Map command to follower state */
	UFUNCTION(BlueprintPure, Category = "Follower|State")
	static EFollowerState MapCommandToState(EStrategicCommandType CommandType);

	/** Transition to new follower state */
	UFUNCTION(BlueprintCallable, Category = "Follower|State")
	void TransitionToState(EFollowerState NewState);

	/** Get current follower state */
	UFUNCTION(BlueprintPure, Category = "Follower|State")
	EFollowerState GetCurrentState() const { return CurrentFollowerState; }

	/** Mark follower as dead */
	UFUNCTION(BlueprintCallable, Category = "Follower|State")
	void MarkAsDead();

	/** Mark follower as alive (e.g., respawn) */
	UFUNCTION(BlueprintCallable, Category = "Follower|State")
	void MarkAsAlive();

	//--------------------------------------------------------------------------
	// OBSERVATION
	//--------------------------------------------------------------------------

	/** Update local observation */
	UFUNCTION(BlueprintCallable, Category = "Follower|Observation")
	void UpdateLocalObservation(const FObservationElement& NewObservation);

	/** Get local observation */
	UFUNCTION(BlueprintPure, Category = "Follower|Observation")
	FObservationElement GetLocalObservation() const { return LocalObservation; }

	/** Build observation from current actor state */
	UFUNCTION(BlueprintCallable, Category = "Follower|Observation")
	FObservationElement BuildLocalObservation();

	//--------------------------------------------------------------------------
	// REINFORCEMENT LEARNING
	//--------------------------------------------------------------------------

	/** Query RL policy for tactical action */
	UFUNCTION(BlueprintCallable, Category = "Follower|RL")
	ETacticalAction QueryRLPolicy();

	/** Get action probabilities from RL policy */
	UFUNCTION(BlueprintCallable, Category = "Follower|RL")
	TArray<float> GetRLActionProbabilities();

	/** Provide reward feedback to RL policy */
	UFUNCTION(BlueprintCallable, Category = "Follower|RL")
	void ProvideReward(float Reward, bool bTerminal = false);

	/** Accumulate reward (alias for ProvideReward for compatibility) */
	UFUNCTION(BlueprintCallable, Category = "Follower|RL")
	void AccumulateReward(float Reward) { ProvideReward(Reward, false); }

	/** Get accumulated reward this episode */
	UFUNCTION(BlueprintPure, Category = "Follower|RL")
	float GetAccumulatedReward() const { return AccumulatedReward; }

	/** Reset episode (clear accumulated reward) */
	UFUNCTION(BlueprintCallable, Category = "Follower|RL")
	void ResetEpisode();

	/** Export collected experiences to JSON */
	UFUNCTION(BlueprintCallable, Category = "Follower|RL")
	bool ExportExperiences(const FString& FilePath);

	/** Is tactical policy ready for queries? */
	UFUNCTION(BlueprintPure, Category = "Follower|RL")
	bool IsTacticalPolicyReady() const { return TacticalPolicy != nullptr; }

	/** Get tactical policy (nullptr if not set) */
	UFUNCTION(BlueprintPure, Category = "Follower|RL")
	URLPolicyNetwork* GetTacticalPolicy() const { return TacticalPolicy; }

	//--------------------------------------------------------------------------
	// UTILITY
	//--------------------------------------------------------------------------

	/** Get team leader */
	UFUNCTION(BlueprintPure, Category = "Follower|Team")
	UTeamLeaderComponent* GetTeamLeader() const { return TeamLeader; }

	/** Is registered with team leader? */
	UFUNCTION(BlueprintPure, Category = "Follower|Team")
	bool IsRegisteredWithLeader() const;

	/** Draw debug info */
	UFUNCTION(BlueprintCallable, Category = "Follower|Debug")
	void DrawDebugInfo();

	/** Get state name as string */
	UFUNCTION(BlueprintPure, Category = "Follower|State")
	static FString GetStateName(EFollowerState State);

private:
	/** Initialize state machine */
	void InitializeStateMachine();

	/** Update command timer */
	void UpdateCommandTimer(float DeltaTime);

	/** Previous state (for state change detection) */
	EFollowerState PreviousFollowerState = EFollowerState::Idle;
};
