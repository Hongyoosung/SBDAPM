#include "Team/FollowerAgentComponent.h"
#include "Team/TeamLeaderComponent.h"
#include "RL/RLPolicyNetwork.h"
#include "Perception/AgentPerceptionComponent.h"
#include "Combat/HealthComponent.h"
#include "Combat/WeaponComponent.h"
#include "Core/SimulationManagerGameMode.h"
#include "DrawDebugHelpers.h"
#include "AIController.h"
#include "Kismet/GameplayStatics.h"
#include "StateTree/FollowerStateTreeComponent.h"

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

	// Bind to HealthComponent events for RL reward integration
	UHealthComponent* HealthComp = GetOwner()->FindComponentByClass<UHealthComponent>();
	if (HealthComp)
	{
		HealthComp->OnDamageTaken.AddDynamic(this, &UFollowerAgentComponent::OnDamageTakenEvent);
		HealthComp->OnDamageDealt.AddDynamic(this, &UFollowerAgentComponent::OnDamageDealtEvent);
		HealthComp->OnKillConfirmed.AddDynamic(this, &UFollowerAgentComponent::OnKillEvent);
		HealthComp->OnDeath.AddDynamic(this, &UFollowerAgentComponent::OnDeathEvent);

		UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Bound to HealthComponent events for RL rewards"),
			*GetOwner()->GetName());
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("FollowerAgent '%s': No HealthComponent found, RL combat rewards disabled"),
			*GetOwner()->GetName());
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
		UE_LOG(LogTemp, Error, TEXT("[FOLLOWER] '%s': ❌ Cannot signal event, no TeamLeader!"),
			*GetOwner()->GetName());
		return;
	}

	// Use owner's location if not specified
	if (Location.IsZero() && GetOwner())
	{
		Location = GetOwner()->GetActorLocation();
	}

	FString EventName = UEnum::GetValueAsString(Event);
	FString InstigatorName = Instigator ? Instigator->GetName() : TEXT("None");

	UE_LOG(LogTemp, Warning, TEXT("[FOLLOWER] '%s': 📡 Signaling event '%s' to Team Leader '%s' (Instigator: %s, Priority: %d)"),
		*GetOwner()->GetName(),
		*EventName,
		*TeamLeader->TeamName,
		*InstigatorName,
		Priority);

	TeamLeader->ProcessStrategicEvent(Event, Instigator, Location, Priority);

	UE_LOG(LogTemp, Display, TEXT("[FOLLOWER] '%s': ✅ Event signaled successfully"),
		*GetOwner()->GetName());

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

	FString TargetInfo = Command.TargetActor ?
		FString::Printf(TEXT("Target: %s"), *Command.TargetActor->GetName()) :
		(!Command.TargetLocation.IsZero() ?
			FString::Printf(TEXT("Location: %s"), *Command.TargetLocation.ToCompactString()) :
			TEXT("No target"));

	UE_LOG(LogTemp, Warning, TEXT("[COMMAND RECEIVED] '%s': 📥 Executing command '%s' → State '%s'"),
		*GetOwner()->GetName(),
		*UEnum::GetValueAsString(Command.CommandType),
		*GetStateName(NewState));

	UE_LOG(LogTemp, Display, TEXT("[COMMAND RECEIVED] '%s':    Priority: %d, %s, Duration: %.1fs"),
		*GetOwner()->GetName(),
		Command.Priority,
		*TargetInfo,
		Command.ExpectedDuration);

	// Transition FSM
	TransitionToState(NewState);

	UE_LOG(LogTemp, Warning, TEXT("[COMMAND RECEIVED] '%s': ✅ Command execution initiated"),
		*GetOwner()->GetName());

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

	UE_LOG(LogTemp, Warning, TEXT("FollowerAgent '%s': Respawning - resetting health and systems"), *GetOwner()->GetName());

	// Reset health to full
	UHealthComponent* HealthComp = GetOwner()->FindComponentByClass<UHealthComponent>();
	if (HealthComp)
	{
		HealthComp->ResetHealth();
		UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Health reset to %.1f/%.1f"),
			*GetOwner()->GetName(), HealthComp->GetCurrentHealth(), HealthComp->GetMaxHealth());
	}

	// Reset episode and clear old commands
	ResetEpisode();
	CurrentCommand = FStrategicCommand(); // Clear old command
	TimeSinceLastCommand = 0.0f;

	// Transition to Idle state
	TransitionToState(EFollowerState::Idle);

	// Notify StateTree of respawn (this will exit Dead state and re-enable systems)
	if (UFollowerStateTreeComponent* StateTreeComp = GetOwner()->FindComponentByClass<UFollowerStateTreeComponent>())
	{
		StateTreeComp->OnFollowerRespawned();
	}

	UE_LOG(LogTemp, Warning, TEXT("FollowerAgent '%s': Respawn complete - ready for new commands"), *GetOwner()->GetName());
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

	AActor* Owner = GetOwner();
	if (!Owner) return Observation;

	// Basic agent state
	Observation.Position = Owner->GetActorLocation();
	Observation.Velocity = Owner->GetVelocity();
	Observation.Rotation = Owner->GetActorRotation();

	// Get perception component
	UAgentPerceptionComponent* PerceptionComp = Owner->FindComponentByClass<UAgentPerceptionComponent>();
	if (PerceptionComp)
	{
		// Update enemy information from perception
		PerceptionComp->UpdateObservationWithEnemies(Observation);

		// Build raycast hit types (16 rays at 5000cm range)
		Observation.RaycastHitTypes = PerceptionComp->BuildRaycastHitTypes(16, 5000.0f);

		// Calculate raycast distances (normalized)
		const FVector OwnerLocation = Owner->GetActorLocation();
		Observation.RaycastDistances.Init(1.0f, 16);

		FCollisionQueryParams QueryParams;
		QueryParams.AddIgnoredActor(Owner);

		const float MaxRayDistance = 5000.0f;
		const float AngleStep = 360.0f / 16;

		for (int32 i = 0; i < 16; ++i)
		{
			const float Angle = i * AngleStep;
			const FRotator RayRotation = Observation.Rotation + FRotator(0, Angle, 0);
			const FVector RayDirection = RayRotation.Vector();
			const FVector EndLocation = OwnerLocation + (RayDirection * MaxRayDistance);

			FHitResult HitResult;
			if (GetWorld()->LineTraceSingleByChannel(HitResult, OwnerLocation, EndLocation,
				ECC_Visibility, QueryParams))
			{
				// Normalized distance (0-1)
				Observation.RaycastDistances[i] = FMath::Clamp(HitResult.Distance / MaxRayDistance, 0.0f, 1.0f);
			}
		}
	}
	else
	{
		// Initialize empty if no perception component
		Observation.InitializeRaycasts(16);
	}

	// Gather combat state from components
	UHealthComponent* HealthComp = Owner->FindComponentByClass<UHealthComponent>();
	if (HealthComp)
	{
		Observation.AgentHealth = HealthComp->GetHealthPercentage() * 100.0f;
		Observation.Shield = HealthComp->GetArmor(); // Map armor to shield
	}

	UWeaponComponent* WeaponComp = Owner->FindComponentByClass<UWeaponComponent>();
	if (WeaponComp)
	{
		Observation.WeaponCooldown = WeaponComp->GetRemainingCooldown();
		Observation.Ammunition = WeaponComp->HasAmmo() ?
			(float)WeaponComp->GetCurrentAmmo() / FMath::Max(WeaponComp->GetMaxAmmo(), 1) * 100.0f : 0.0f;
	}

	return Observation;
}

//------------------------------------------------------------------------------
// REINFORCEMENT LEARNING
//------------------------------------------------------------------------------


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

	/*UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Provided reward %.2f (Accumulated: %.2f, Terminal: %s)"),
		*GetOwner()->GetName(), Reward, AccumulatedReward, bTerminal ? TEXT("Yes") : TEXT("No"));*/

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

void UFollowerAgentComponent::ClearExperiences()
{
	if (TacticalPolicy)
	{
		TacticalPolicy->ClearExperiences();
		UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Cleared experience buffer"), *GetOwner()->GetName());
	}
}

void UFollowerAgentComponent::OnEpisodeEnded(float EpisodeReward)
{
	// Add episode reward to accumulated reward
	AccumulatedReward += EpisodeReward;

	// Provide terminal reward to RL policy
	ProvideReward(EpisodeReward, true); // bTerminal = true

	UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Episode ended - EpisodeReward=%.2f, TotalAccumulated=%.2f"),
		*GetOwner()->GetName(), EpisodeReward, AccumulatedReward);
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

	// Draw line to command target (actor or location)
	if (IsCommandValid())
	{
		// Check if target actor exists and is alive
		if (CurrentCommand.TargetActor && IsValid(CurrentCommand.TargetActor))
		{
			// Check if target has health component and is alive
			UHealthComponent* TargetHealth = CurrentCommand.TargetActor->FindComponentByClass<UHealthComponent>();
			bool bTargetAlive = !TargetHealth || TargetHealth->IsAlive();

			if (bTargetAlive)
			{
				FVector TargetPos = CurrentCommand.TargetActor->GetActorLocation();
				DrawDebugLine(World, FollowerPos, TargetPos, FColor::Red, false, 0.1f, 0, 2.0f);
				DrawDebugSphere(World, TargetPos, 50.0f, 8, FColor::Red, false, 0.1f);

				// Draw target name
				FString TargetName = FString::Printf(TEXT("Target: %s"), *CurrentCommand.TargetActor->GetName());
				DrawDebugString(World, TargetPos + FVector(0, 0, 100), TargetName, nullptr, FColor::Red, 0.1f, true);
			}
		}
		// Draw to target location if no valid actor
		else if (!CurrentCommand.TargetLocation.IsZero())
		{
			DrawDebugLine(World, FollowerPos, CurrentCommand.TargetLocation, FColor::Yellow, false, 0.1f, 0, 2.0f);
			DrawDebugSphere(World, CurrentCommand.TargetLocation, 50.0f, 8, FColor::Yellow, false, 0.1f);
		}
	}

	// Draw line to team leader
	if (TeamLeader && TeamLeader->GetOwner())
	{
		FVector LeaderPos = TeamLeader->GetOwner()->GetActorLocation();
		DrawDebugLine(World, FollowerPos, LeaderPos, TeamLeader->TeamColor.ToFColor(true), false, 0.1f, 0, 1.0f);
	}
}

//------------------------------------------------------------------------------
// COMBAT EVENT HANDLERS (RL REWARD INTEGRATION)
//------------------------------------------------------------------------------

void UFollowerAgentComponent::OnDamageTakenEvent(const FDamageEventData& DamageEvent, float CurrentHealth)
{
	// Provide negative reward for taking damage
	ProvideReward(FTacticalRewards::TAKE_DAMAGE, false);

	UE_LOG(LogTemp, Verbose, TEXT("FollowerAgent '%s': Took %.1f damage from %s (Reward: %.1f)"),
		*GetOwner()->GetName(),
		DamageEvent.DamageAmount,
		DamageEvent.Instigator ? *DamageEvent.Instigator->GetName() : TEXT("Unknown"),
		FTacticalRewards::TAKE_DAMAGE);

	// Signal event to team leader if damage is significant
	if (TeamLeader && DamageEvent.DamageAmount >= 10.0f)
	{
		SignalEventToLeader(EStrategicEvent::UnderFire, DamageEvent.Instigator, GetOwner()->GetActorLocation(), 6);
	}
}

void UFollowerAgentComponent::OnDamageDealtEvent(AActor* Victim, float DamageAmount)
{
	// Provide positive reward for dealing damage
	ProvideReward(FTacticalRewards::DAMAGE_ENEMY, false);

	UE_LOG(LogTemp, Verbose, TEXT("FollowerAgent '%s': Dealt %.1f damage to %s (Reward: %.1f)"),
		*GetOwner()->GetName(),
		DamageAmount,
		Victim ? *Victim->GetName() : TEXT("Unknown"),
		FTacticalRewards::DAMAGE_ENEMY);
}

void UFollowerAgentComponent::OnKillEvent(AActor* Victim, float TotalDamage)
{
	// Provide large positive reward for kill
	ProvideReward(FTacticalRewards::KILL_ENEMY, false);

	UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': KILL confirmed on %s (Reward: %.1f)"),
		*GetOwner()->GetName(),
		Victim ? *Victim->GetName() : TEXT("Unknown"),
		FTacticalRewards::KILL_ENEMY);

	// Signal kill to team leader
	if (TeamLeader)
	{
		SignalEventToLeader(EStrategicEvent::EnemyKilled, Victim, GetOwner()->GetActorLocation(), 7);
	}
}

void UFollowerAgentComponent::OnDeathEvent(const FDeathEventData& DeathEvent)
{
	// Provide large negative reward for death (terminal state)
	ProvideReward(FTacticalRewards::DIE, true);

	UE_LOG(LogTemp, Warning, TEXT("FollowerAgent '%s': DIED (Killed by %s, Reward: %.1f)"),
		*GetOwner()->GetName(),
		DeathEvent.Killer ? *DeathEvent.Killer->GetName() : TEXT("Unknown"),
		FTacticalRewards::DIE);

	// Mark as dead
	MarkAsDead();

	// Signal death to team leader
	if (TeamLeader)
	{
		SignalEventToLeader(EStrategicEvent::AllyKilled, DeathEvent.Killer, GetOwner()->GetActorLocation(), 10);
	}
}
