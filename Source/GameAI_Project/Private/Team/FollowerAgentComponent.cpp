#include "Team/FollowerAgentComponent.h"
#include "Team/TeamLeaderComponent.h"
#include "RL/RLPolicyNetwork.h"
#include "DrawDebugHelpers.h"
#include "AIController.h"
#include "Kismet/GameplayStatics.h"

UFollowerAgentComponent::UFollowerAgentComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.TickInterval = 0.1f;  // Update every 100ms
}

void UFollowerAgentComponent::BeginPlay()
{
	Super::BeginPlay();

	// Initialize RL policy if not set
	if (!TacticalPolicy && bUseRLPolicy)
	{
		TacticalPolicy = NewObject<URLPolicyNetwork>(this);
		FRLPolicyConfig Config;
		TacticalPolicy->Initialize(Config);
		TacticalPolicy->bCollectExperiences = bCollectExperiences;

		UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Created RL policy"), *GetOwner()->GetName());
	}

	// Find team leader component
	if (!TeamLeader)
	{
		// Option 1: Get from TeamLeaderActor if specified
		if (TeamLeaderActor)
		{
			TeamLeader = TeamLeaderActor->FindComponentByClass<UTeamLeaderComponent>();
			if (TeamLeader)
			{
				UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Found TeamLeader on specified actor '%s'"),
					*GetOwner()->GetName(), *TeamLeaderActor->GetName());
			}
			else
			{
				UE_LOG(LogTemp, Error, TEXT("FollowerAgent '%s': TeamLeaderActor '%s' has no TeamLeaderComponent!"),
					*GetOwner()->GetName(), *TeamLeaderActor->GetName());
			}
		}
		// Option 2: Auto-find by tag
		else if (TeamLeaderTag != NAME_None)
		{
			TArray<AActor*> FoundActors;
			UGameplayStatics::GetAllActorsWithTag(GetWorld(), TeamLeaderTag, FoundActors);

			if (FoundActors.Num() > 0)
			{
				TeamLeaderActor = FoundActors[0];
				TeamLeader = TeamLeaderActor->FindComponentByClass<UTeamLeaderComponent>();

				if (TeamLeader)
				{
					UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Auto-found TeamLeader on actor '%s' by tag '%s'"),
						*GetOwner()->GetName(), *TeamLeaderActor->GetName(), *TeamLeaderTag.ToString());
				}
				else
				{
					UE_LOG(LogTemp, Warning, TEXT("FollowerAgent '%s': Found actor with tag '%s' but no TeamLeaderComponent"),
						*GetOwner()->GetName(), *TeamLeaderTag.ToString());
				}
			}
			else
			{
				UE_LOG(LogTemp, Warning, TEXT("FollowerAgent '%s': No actor found with tag '%s'"),
					*GetOwner()->GetName(), *TeamLeaderTag.ToString());
			}
		}
	}

	// Auto-register with team leader
	if (bAutoRegisterWithLeader && TeamLeader)
	{
		RegisterWithTeamLeader();
	}

	UE_LOG(LogTemp, Log, TEXT("FollowerAgentComponent: Initialized on %s"), *GetOwner()->GetName());
}

void UFollowerAgentComponent::TickComponent(float DeltaTime, ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// Update command timer
	UpdateCommandTimer(DeltaTime);

	// Draw debug info if enabled
	if (bEnableDebugDrawing)
	{
		DrawDebugInfo();
	}
}

void UFollowerAgentComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	// Unregister from team leader
	UnregisterFromTeamLeader();

	Super::EndPlay(EndPlayReason);
}

//------------------------------------------------------------------------------
// INITIALIZATION
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// TEAM LEADER COMMUNICATION
//------------------------------------------------------------------------------

bool UFollowerAgentComponent::RegisterWithTeamLeader()
{
	if (!TeamLeader)
	{
		UE_LOG(LogTemp, Warning, TEXT("FollowerAgent '%s': No TeamLeader set, cannot register"),
			*GetOwner()->GetName());
		return false;
	}

	bool bSuccess = TeamLeader->RegisterFollower(GetOwner());

	if (bSuccess)
	{
		UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Registered with TeamLeader '%s'"),
			*GetOwner()->GetName(), *TeamLeader->TeamName);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("FollowerAgent '%s': Failed to register with TeamLeader"),
			*GetOwner()->GetName());
	}

	return bSuccess;
}

void UFollowerAgentComponent::UnregisterFromTeamLeader()
{
	if (!TeamLeader) return;

	if (IsRegisteredWithLeader())
	{
		TeamLeader->UnregisterFollower(GetOwner());
		UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Unregistered from TeamLeader"),
			*GetOwner()->GetName());
	}
}

void UFollowerAgentComponent::SignalEventToLeader(
	EStrategicEvent Event,
	AActor* Instigator,
	FVector Location,
	int32 Priority)
{
	if (!TeamLeader)
	{
		UE_LOG(LogTemp, Warning, TEXT("FollowerAgent '%s': Cannot signal event, no TeamLeader"),
			*GetOwner()->GetName());
		return;
	}

	// Use owner's location if not specified
	if (Location.IsZero() && GetOwner())
	{
		Location = GetOwner()->GetActorLocation();
	}

	TeamLeader->ProcessStrategicEvent(Event, Instigator, Location, Priority);

	UE_LOG(LogTemp, Verbose, TEXT("FollowerAgent '%s': Signaled event %d to TeamLeader (Priority: %d)"),
		*GetOwner()->GetName(), static_cast<int32>(Event), Priority);

	// Broadcast event
	OnEventSignaled.Broadcast(Event, Instigator, Priority);
}

void UFollowerAgentComponent::ReportCommandComplete(bool bSuccess)
{
	if (!IsCommandValid()) return;

	CurrentCommand.bCompleted = true;
	CurrentCommand.Progress = 1.0f;

	UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Command %s completed (%s)"),
		*GetOwner()->GetName(),
		*UEnum::GetValueAsString(CurrentCommand.CommandType),
		bSuccess ? TEXT("Success") : TEXT("Failed"));

	// Signal completion event to leader (low priority)
	if (bSuccess)
	{
		SignalEventToLeader(EStrategicEvent::Custom, GetOwner(), FVector::ZeroVector, 2);
	}
}

void UFollowerAgentComponent::RequestAssistance(int32 Priority)
{
	SignalEventToLeader(EStrategicEvent::AllyRescueSignal, GetOwner(), FVector::ZeroVector, Priority);

	UE_LOG(LogTemp, Warning, TEXT("FollowerAgent '%s': Requested assistance (Priority: %d)"),
		*GetOwner()->GetName(), Priority);
}

//------------------------------------------------------------------------------
// COMMAND EXECUTION
//------------------------------------------------------------------------------

void UFollowerAgentComponent::ExecuteCommand(const FStrategicCommand& Command)
{
	// Store command
	CurrentCommand = Command;
	TimeSinceLastCommand = 0.0f;

	// Map command to follower state
	EFollowerState NewState = MapCommandToState(Command.CommandType);

	UE_LOG(LogTemp, Warning, TEXT("🟢 FollowerAgent '%s': Executing command %s → State %s"),
		*GetOwner()->GetName(),
		*UEnum::GetValueAsString(Command.CommandType),
		*GetStateName(NewState));

	// Transition FSM
	TransitionToState(NewState);

	// Broadcast event
	OnCommandReceived.Broadcast(Command, NewState);
}

bool UFollowerAgentComponent::IsCommandValid() const
{
	// Command is valid if not completed and not idle
	return !CurrentCommand.bCompleted && CurrentCommand.CommandType != EStrategicCommandType::Idle;
}

bool UFollowerAgentComponent::HasActiveCommand() const
{
	// Has active command if not idle and command is valid
	return CurrentCommand.CommandType != EStrategicCommandType::Idle && IsCommandValid();
}

void UFollowerAgentComponent::UpdateCommandProgress(float Progress)
{
	CurrentCommand.Progress = FMath::Clamp(Progress, 0.0f, 1.0f);

	UE_LOG(LogTemp, Verbose, TEXT("FollowerAgent '%s': Command progress %.1f%%"),
		*GetOwner()->GetName(), CurrentCommand.Progress * 100.0f);
}

//------------------------------------------------------------------------------
// STATE MANAGEMENT
//------------------------------------------------------------------------------

EFollowerState UFollowerAgentComponent::MapCommandToState(EStrategicCommandType CommandType)
{
	switch (CommandType)
	{
		// Offensive
		case EStrategicCommandType::Assault:
			return EFollowerState::Assault;

		// Defensive
		case EStrategicCommandType::Defend:
			return EFollowerState::Defend;

		// Support
		case EStrategicCommandType::Support:
			return EFollowerState::Support;

		// Movement
		case EStrategicCommandType::MoveTo:
			return EFollowerState::Move;

		// Retreat
		case EStrategicCommandType::Retreat:
			return EFollowerState::Retreat;

		// Idle
		case EStrategicCommandType::Idle:
		default:
			return EFollowerState::Idle;
	}
}

void UFollowerAgentComponent::TransitionToState(EFollowerState NewState)
{
	if (CurrentFollowerState == NewState)
	{
		UE_LOG(LogTemp, Verbose, TEXT("FollowerAgent '%s': Already in state %s"),
			*GetOwner()->GetName(), *GetStateName(NewState));
		return;
	}

	EFollowerState OldState = CurrentFollowerState;
	CurrentFollowerState = NewState;

	UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': State transition %s → %s"),
		*GetOwner()->GetName(),
		*GetStateName(OldState),
		*GetStateName(NewState));

	// Broadcast state change event
	OnStateChanged.Broadcast(OldState, NewState);

	// TODO: Transition existing FSM to corresponding state
	// This would require mapping EFollowerState to existing UState classes
}

void UFollowerAgentComponent::MarkAsDead()
{
	if (!bIsAlive) return;

	bIsAlive = false;
	TransitionToState(EFollowerState::Dead);

	// Signal death to team leader
	SignalEventToLeader(EStrategicEvent::AllyKilled, GetOwner(), FVector::ZeroVector, 10);

	UE_LOG(LogTemp, Warning, TEXT("FollowerAgent '%s': Marked as dead"), *GetOwner()->GetName());
}

void UFollowerAgentComponent::MarkAsAlive()
{
	if (bIsAlive) return;

	bIsAlive = true;
	TransitionToState(EFollowerState::Idle);

	UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Marked as alive"), *GetOwner()->GetName());
}

//------------------------------------------------------------------------------
// OBSERVATION
//------------------------------------------------------------------------------

void UFollowerAgentComponent::UpdateLocalObservation(const FObservationElement& NewObservation)
{
	LocalObservation = NewObservation;
}

FObservationElement UFollowerAgentComponent::BuildLocalObservation()
{
	FObservationElement Observation;

	if (GetOwner())
	{
		// Basic agent state
		Observation.Position = GetOwner()->GetActorLocation();
		Observation.Velocity = GetOwner()->GetVelocity();
		Observation.Rotation = GetOwner()->GetActorRotation();

		// TODO: Gather combat state, perception, enemies, etc.
		// This should be implemented in BTService_UpdateObservation
	}

	return Observation;
}

//------------------------------------------------------------------------------
// REINFORCEMENT LEARNING
//------------------------------------------------------------------------------

ETacticalAction UFollowerAgentComponent::QueryRLPolicy()
{
	if (!bUseRLPolicy || !TacticalPolicy)
	{
		UE_LOG(LogTemp, Warning, TEXT("FollowerAgent '%s': RL policy not available, returning default action"),
			*GetOwner()->GetName());
		return ETacticalAction::DefensiveHold;
	}

	// Get current observation
	FObservationElement CurrentObs = GetLocalObservation();

	// Query policy for action
	ETacticalAction SelectedAction = TacticalPolicy->SelectAction(CurrentObs);

	// Store for experience collection
	PreviousObservation = CurrentObs;
	LastTacticalAction = SelectedAction;
	TimeSinceLastTacticalAction = 0.0f;  // Reset timer when new action is selected

	UE_LOG(LogTemp, Verbose, TEXT("FollowerAgent '%s': RL policy selected action: %s"),
		*GetOwner()->GetName(), *URLPolicyNetwork::GetActionName(SelectedAction));

	return SelectedAction;
}

TArray<float> UFollowerAgentComponent::GetRLActionProbabilities()
{
	if (!bUseRLPolicy || !TacticalPolicy)
	{
		TArray<float> EmptyProbs;
		return EmptyProbs;
	}

	FObservationElement CurrentObs = GetLocalObservation();
	return TacticalPolicy->GetActionProbabilities(CurrentObs);
}

void UFollowerAgentComponent::ProvideReward(float Reward, bool bTerminal)
{
	// Always accumulate reward (independent of RL policy or experience collection)
	AccumulatedReward += Reward;

	UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Provided reward %.2f (Accumulated: %.2f, Terminal: %s)"),
		*GetOwner()->GetName(), Reward, AccumulatedReward, bTerminal ? TEXT("Yes") : TEXT("No"));

	// Store experience if RL policy is enabled and collecting experiences
	if (bUseRLPolicy && TacticalPolicy && bCollectExperiences)
	{
		FObservationElement CurrentObs = GetLocalObservation();
		TacticalPolicy->StoreExperience(PreviousObservation, LastTacticalAction, Reward, CurrentObs, bTerminal);
	}

	// Reset episode if terminal
	if (bTerminal)
	{
		ResetEpisode();
	}
}

void UFollowerAgentComponent::ResetEpisode()
{
	AccumulatedReward = 0.0f;
	PreviousObservation = FObservationElement();
	LastTacticalAction = ETacticalAction::DefensiveHold;

	UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Episode reset"), *GetOwner()->GetName());
}

bool UFollowerAgentComponent::ExportExperiences(const FString& FilePath)
{
	if (!TacticalPolicy)
	{
		UE_LOG(LogTemp, Error, TEXT("FollowerAgent '%s': No RL policy to export from"), *GetOwner()->GetName());
		return false;
	}

	bool bSuccess = TacticalPolicy->ExportExperiencesToJSON(FilePath);

	if (bSuccess)
	{
		UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Exported %d experiences to %s"),
			*GetOwner()->GetName(), TacticalPolicy->GetExperienceCount(), *FilePath);
	}

	return bSuccess;
}

//------------------------------------------------------------------------------
// UTILITY
//------------------------------------------------------------------------------

bool UFollowerAgentComponent::IsRegisteredWithLeader() const
{
	if (!TeamLeader) return false;

	return TeamLeader->IsFollowerRegistered(GetOwner());
}

FString UFollowerAgentComponent::GetStateName(EFollowerState State)
{
	return UEnum::GetValueAsString(State);
}

void UFollowerAgentComponent::UpdateCommandTimer(float DeltaTime)
{
	TimeSinceLastCommand += DeltaTime;
	TimeSinceLastTacticalAction += DeltaTime;

	// Check for command expiration
	if (IsCommandValid())
	{
		if (CurrentCommand.ExpectedDuration > 0.0f &&
			TimeSinceLastCommand > CurrentCommand.ExpectedDuration)
		{
			UE_LOG(LogTemp, Warning, TEXT("FollowerAgent '%s': Command %s expired after %.1fs"),
				*GetOwner()->GetName(),
				*UEnum::GetValueAsString(CurrentCommand.CommandType),
				TimeSinceLastCommand);

			ReportCommandComplete(false);
		}
	}
}

//------------------------------------------------------------------------------
// DEBUG VISUALIZATION
//------------------------------------------------------------------------------

void UFollowerAgentComponent::DrawDebugInfo()
{
	if (!GetOwner()) return;

	UWorld* World = GetWorld();
	if (!World) return;

	FVector FollowerPos = GetOwner()->GetActorLocation();

	// Draw state above follower
	FString StateText = FString::Printf(TEXT("State: %s\nCommand: %s\nProgress: %.1f%%"),
		*GetStateName(CurrentFollowerState),
		*UEnum::GetValueAsString(CurrentCommand.CommandType),
		CurrentCommand.Progress * 100.0f);

	DrawDebugString(World, FollowerPos + FVector(0, 0, 120), StateText, nullptr, FColor::Cyan, 0.1f, true);

	// Draw line to command target location
	if (IsCommandValid() && !CurrentCommand.TargetLocation.IsZero())
	{
		DrawDebugLine(World, FollowerPos, CurrentCommand.TargetLocation, FColor::Yellow, false, 0.1f, 0, 2.0f);
		DrawDebugSphere(World, CurrentCommand.TargetLocation, 50.0f, 8, FColor::Yellow, false, 0.1f);
	}

	// Draw line to team leader
	if (TeamLeader && TeamLeader->GetOwner())
	{
		FVector LeaderPos = TeamLeader->GetOwner()->GetActorLocation();
		DrawDebugLine(World, FollowerPos, LeaderPos, TeamLeader->TeamColor.ToFColor(true), false, 0.1f, 0, 1.0f);
	}
}
