#include "Team/FollowerAgentComponent.h"
#include "Team/TeamLeaderComponent.h"
#include "RL/RLPolicyNetwork.h"
#include "RL/RewardCalculator.h"
#include "Perception/AgentPerceptionComponent.h"
#include "Combat/HealthComponent.h"
#include "Combat/WeaponComponent.h"
#include "Core/SimulationManagerGameMode.h"
#include "DrawDebugHelpers.h"
#include "AIController.h"
#include "Kismet/GameplayStatics.h"
#include "StateTree/FollowerStateTreeComponent.h"
#include "Team/Objective.h"
#include "Simulation/StateTransition.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

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

	// Find or create RewardCalculator (Sprint 4)
	RewardCalculator = GetOwner()->FindComponentByClass<URewardCalculator>();
	if (!RewardCalculator)
	{
		UE_LOG(LogTemp, Warning, TEXT("FollowerAgent '%s': No RewardCalculator found, hierarchical rewards disabled"),
			*GetOwner()->GetName());
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Found RewardCalculator, hierarchical rewards enabled"),
			*GetOwner()->GetName());
	}

	UE_LOG(LogTemp, Log, TEXT("FollowerAgentComponent: Initialized on %s"), *GetOwner()->GetName());
}

void UFollowerAgentComponent::TickComponent(float DeltaTime, ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// Update tactical action timer (v3.0)
	TimeSinceLastTacticalAction += DeltaTime;

	// Calculate hierarchical rewards if RewardCalculator is available
	if (RewardCalculator)
	{
		float TotalReward = RewardCalculator->CalculateTotalReward(DeltaTime);
		if (FMath::Abs(TotalReward) > 0.01f) // Only provide non-zero rewards
		{
			ProvideReward(TotalReward, false);
		}
	}

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

void UFollowerAgentComponent::ReportObjectiveComplete(bool bSuccess)
{
	if (!HasActiveObjective()) return;

	// Mark objective as completed or failed
	if (bSuccess)
	{
		CurrentObjective->Status = EObjectiveStatus::Completed;
		CurrentObjective->Progress = 1.0f;
	}
	else
	{
		CurrentObjective->Status = EObjectiveStatus::Failed;
	}

	UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Objective %s completed (%s)"),
		*GetOwner()->GetName(),
		*UEnum::GetValueAsString(CurrentObjective->Type),
		bSuccess ? TEXT("Success") : TEXT("Failed"));

	// Signal completion event to leader (low priority)
	if (bSuccess)
	{
		SignalEventToLeader(EStrategicEvent::ObjectiveComplete, GetOwner(), FVector::ZeroVector, 2);
	}
	else
	{
		SignalEventToLeader(EStrategicEvent::ObjectiveFailed, GetOwner(), FVector::ZeroVector, 2);
	}
}

void UFollowerAgentComponent::RequestAssistance(int32 Priority)
{
	SignalEventToLeader(EStrategicEvent::AllyRescueSignal, GetOwner(), FVector::ZeroVector, Priority);

	UE_LOG(LogTemp, Warning, TEXT("FollowerAgent '%s': Requested assistance (Priority: %d)"),
		*GetOwner()->GetName(), Priority);
}

//------------------------------------------------------------------------------
// OBJECTIVE EXECUTION (v3.0)
//------------------------------------------------------------------------------

bool UFollowerAgentComponent::HasActiveObjective() const
{
	return CurrentObjective != nullptr && CurrentObjective->IsActive();
}

//------------------------------------------------------------------------------
// STATE MANAGEMENT
//------------------------------------------------------------------------------

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

	// Broadcast state change event (StateTree reacts via evaluators/conditions)
	OnStateChanged.Broadcast(OldState, NewState);

	// State changes are automatically handled by StateTree evaluators and conditions
	// No explicit FSM transition needed - StateTree evaluates conditions each tick
	// and transitions based on current objective and alive status
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

	// Reset episode (v3.0: objectives are cleared via ObjectiveManager)
	ResetEpisode();

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
	// v3.0: No longer using discrete action probabilities, uses atomic actions instead
	// Return empty array for backward compatibility
	TArray<float> EmptyProbs;
	return EmptyProbs;
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


//------------------------------------------------------------------------------
// DEBUG VISUALIZATION
//------------------------------------------------------------------------------

void UFollowerAgentComponent::DrawDebugInfo()
{
	if (!GetOwner()) return;

	UWorld* World = GetWorld();
	if (!World) return;

	FVector FollowerPos = GetOwner()->GetActorLocation();

	// Draw state above follower (v3.0)
	FString ObjectiveStr = CurrentObjective ? UEnum::GetValueAsString(CurrentObjective->Type) : TEXT("None");
	float Progress = CurrentObjective ? CurrentObjective->GetProgress() : 0.0f;

	FString StateText = FString::Printf(TEXT("State: %s\nObjective: %s\nProgress: %.1f%%"),
		*GetStateName(CurrentFollowerState),
		*ObjectiveStr,
		Progress * 100.0f);

	DrawDebugString(World, FollowerPos + FVector(0, 0, 120), StateText, nullptr, FColor::Cyan, 0.1f, true);

	// Draw line to objective target (actor or location)
	if (HasActiveObjective())
	{
		// Check if target actor exists and is alive
		if (CurrentObjective->TargetActor && IsValid(CurrentObjective->TargetActor))
		{
			// Check if target has health component and is alive
			UHealthComponent* TargetHealth = CurrentObjective->TargetActor->FindComponentByClass<UHealthComponent>();
			bool bTargetAlive = !TargetHealth || TargetHealth->IsAlive();

			if (bTargetAlive)
			{
				FVector TargetPos = CurrentObjective->TargetActor->GetActorLocation();
				DrawDebugLine(World, FollowerPos, TargetPos, FColor::Red, false, 0.1f, 0, 2.0f);
				DrawDebugSphere(World, TargetPos, 50.0f, 8, FColor::Red, false, 0.1f);

				// Draw target name
				FString TargetName = FString::Printf(TEXT("Target: %s"), *CurrentObjective->TargetActor->GetName());
				DrawDebugString(World, TargetPos + FVector(0, 0, 100), TargetName, nullptr, FColor::Red, 0.1f, true);
			}
		}
		// Draw to target location if no valid actor
		else if (!CurrentObjective->TargetLocation.IsZero())
		{
			DrawDebugLine(World, FollowerPos, CurrentObjective->TargetLocation, FColor::Yellow, false, 0.1f, 0, 2.0f);
			DrawDebugSphere(World, CurrentObjective->TargetLocation, 50.0f, 8, FColor::Yellow, false, 0.1f);
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
	// Notify RewardCalculator (Sprint 4)
	if (RewardCalculator)
	{
		RewardCalculator->OnTakeDamage(DamageEvent.DamageAmount);
	}
	else
	{
		// Fallback to direct reward
		ProvideReward(FTacticalRewards::TAKE_DAMAGE, false);
	}

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
	// Notify RewardCalculator (Sprint 4)
	if (RewardCalculator)
	{
		RewardCalculator->OnDealDamage(DamageAmount, Victim);
	}
	else
	{
		// Fallback to direct reward
		ProvideReward(FTacticalRewards::DAMAGE_ENEMY, false);
	}

	UE_LOG(LogTemp, Verbose, TEXT("FollowerAgent '%s': Dealt %.1f damage to %s (Reward: %.1f)"),
		*GetOwner()->GetName(),
		DamageAmount,
		Victim ? *Victim->GetName() : TEXT("Unknown"),
		FTacticalRewards::DAMAGE_ENEMY);
}

void UFollowerAgentComponent::OnKillEvent(AActor* Victim, float TotalDamage)
{
	// Notify RewardCalculator (Sprint 4)
	if (RewardCalculator)
	{
		RewardCalculator->OnKillEnemy(Victim);
	}
	else
	{
		// Fallback to direct reward
		ProvideReward(FTacticalRewards::KILL_ENEMY, false);
	}

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
	// Notify RewardCalculator (Sprint 4)
	if (RewardCalculator)
	{
		RewardCalculator->OnDeath();
	}
	else
	{
		// Fallback to direct reward
		ProvideReward(FTacticalRewards::DIE, true);
	}

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

//------------------------------------------------------------------------------
// OBJECTIVE INTEGRATION (SPRINT 4)
//------------------------------------------------------------------------------

void UFollowerAgentComponent::SetCurrentObjective(UObjective* Objective)
{
	if (RewardCalculator)
	{
		RewardCalculator->SetCurrentObjective(Objective);
		UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Objective set to %s"),
			*GetOwner()->GetName(),
			Objective ? *UEnum::GetValueAsString(Objective->Type) : TEXT("None"));
	}
}

//------------------------------------------------------------------------------
// STATE TRANSITION LOGGING (SPRINT 2 - WORLD MODEL)
//------------------------------------------------------------------------------

void UFollowerAgentComponent::EnableStateTransitionLogging(bool bEnable)
{
	bLogStateTransitions = bEnable;

	if (bEnable)
	{
		LoggedTransitions.Empty();
		LastStateLogTime = 0.0f;
		UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': State transition logging ENABLED"), *GetOwner()->GetName());
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': State transition logging DISABLED (%d samples collected)"),
			*GetOwner()->GetName(), LoggedTransitions.Num());
	}
}

void UFollowerAgentComponent::LogStateTransition()
{
	if (!bLogStateTransitions || !TeamLeader)
	{
		return;
	}

	// Throttle logging (only log at intervals)
	const float CurrentTime = GetWorld()->GetTimeSeconds();
	if (CurrentTime - LastStateLogTime < StateLogInterval)
	{
		return;
	}

	// Get current team observation from leader
	FTeamObservation CurrentTeamObs = TeamLeader->CurrentTeamObservation;

	// Only log if we have a previous observation
	if (PreviousTeamObservation.AliveFollowers > 0)
	{
		FStateTransitionSample Sample;

		// State before
		Sample.StateBefore = PreviousTeamObservation.Flatten();

		// Actions (tactical actions only in v3.0)
		Sample.TacticalActions.Add(LastTacticalAction);

		// State after
		Sample.StateAfter = CurrentTeamObs.Flatten();

		// Calculate actual delta
		FTeamStateDelta ActualDelta;
		ActualDelta.TeamHealthDelta = CurrentTeamObs.AverageTeamHealth - PreviousTeamObservation.AverageTeamHealth;
		ActualDelta.AliveCountDelta = CurrentTeamObs.AliveFollowers - PreviousTeamObservation.AliveFollowers;
		ActualDelta.TeamCohesionDelta = CurrentTeamObs.FormationCoherence - PreviousTeamObservation.FormationCoherence;
		ActualDelta.DeltaTime = StateLogInterval;

		Sample.ActualDelta = ActualDelta;
		Sample.Timestamp = CurrentTime;

		// Store sample
		LoggedTransitions.Add(Sample);

		if (LoggedTransitions.Num() % 50 == 0)
		{
			UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Logged %d state transitions"),
				*GetOwner()->GetName(), LoggedTransitions.Num());
		}
	}

	// Update previous observation
	PreviousTeamObservation = CurrentTeamObs;
	LastStateLogTime = CurrentTime;
}

bool UFollowerAgentComponent::ExportStateTransitions(const FString& FilePath)
{
	if (LoggedTransitions.Num() == 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("FollowerAgent '%s': No state transitions to export"), *GetOwner()->GetName());
		return false;
	}

	// Build JSON
	FString JsonString = TEXT("{\n  \"transitions\": [\n");

	for (int32 i = 0; i < LoggedTransitions.Num(); ++i)
	{
		const FStateTransitionSample& Sample = LoggedTransitions[i];

		JsonString += TEXT("    {\n");
		JsonString += FString::Printf(TEXT("      \"timestamp\": %.2f,\n"), Sample.Timestamp);
		JsonString += FString::Printf(TEXT("      \"game_outcome\": %.2f,\n"), Sample.GameOutcome);

		// State before
		JsonString += TEXT("      \"state_before\": [");
		for (int32 j = 0; j < Sample.StateBefore.Num(); ++j)
		{
			JsonString += FString::Printf(TEXT("%.4f"), Sample.StateBefore[j]);
			if (j < Sample.StateBefore.Num() - 1) JsonString += TEXT(", ");
		}
		JsonString += TEXT("],\n");

		// State after
		JsonString += TEXT("      \"state_after\": [");
		for (int32 j = 0; j < Sample.StateAfter.Num(); ++j)
		{
			JsonString += FString::Printf(TEXT("%.4f"), Sample.StateAfter[j]);
			if (j < Sample.StateAfter.Num() - 1) JsonString += TEXT(", ");
		}
		JsonString += TEXT("],\n");

		// Actual delta
		JsonString += TEXT("      \"actual_delta\": {\n");
		JsonString += FString::Printf(TEXT("        \"team_health_delta\": %.2f,\n"), Sample.ActualDelta.TeamHealthDelta);
		JsonString += FString::Printf(TEXT("        \"alive_count_delta\": %d,\n"), Sample.ActualDelta.AliveCountDelta);
		JsonString += FString::Printf(TEXT("        \"team_cohesion_delta\": %.4f\n"), Sample.ActualDelta.TeamCohesionDelta);
		JsonString += TEXT("      }\n");

		JsonString += TEXT("    }");
		if (i < LoggedTransitions.Num() - 1) JsonString += TEXT(",");
		JsonString += TEXT("\n");
	}

	JsonString += TEXT("  ]\n}\n");

	// Write to file
	const FString FullPath = FPaths::ProjectDir() / FilePath;
	if (FFileHelper::SaveStringToFile(JsonString, *FullPath))
	{
		UE_LOG(LogTemp, Log, TEXT("FollowerAgent '%s': Exported %d transitions to %s"),
			*GetOwner()->GetName(), LoggedTransitions.Num(), *FullPath);
		return true;
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("FollowerAgent '%s': Failed to export transitions to %s"),
			*GetOwner()->GetName(), *FullPath);
		return false;
	}
}
