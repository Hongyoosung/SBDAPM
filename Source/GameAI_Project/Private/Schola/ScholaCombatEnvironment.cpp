// ScholaCombatEnvironment.cpp - Schola environment implementation

#include "Schola/ScholaCombatEnvironment.h"
#include "Schola/ScholaAgentComponent.h"
#include "Schola/FollowerAgentTrainer.h"
#include "Core/SimulationManagerGameMode.h"
#include "Core/ScholaGameInstance.h"
#include "Team/FollowerAgentComponent.h"
#include "Team/TeamLeaderComponent.h"
#include "Communicator/CommunicationManager.h"
#include "Subsystem/ScholaManagerSubsystem.h"
#include "EngineUtils.h"
#include "Kismet/GameplayStatics.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/DefaultPawn.h"

// Static singleton reference
AScholaCombatEnvironment* AScholaCombatEnvironment::PrimaryEnvironmentInstance = nullptr;

AScholaCombatEnvironment::AScholaCombatEnvironment(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	PrimaryActorTick.bCanEverTick = false;
}

void AScholaCombatEnvironment::BeginPlay()
{
	// NOTE: Do NOT call Super::BeginPlay() yet - we need to set up first

	// SINGLETON ENFORCEMENT: Only allow ONE active environment per world
	if (PrimaryEnvironmentInstance == nullptr)
	{
		// This is the first instance - make it primary
		PrimaryEnvironmentInstance = this;
		bIsPrimaryEnvironment = true;
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] %s is PRIMARY environment"), *GetName());
	}
	else if (PrimaryEnvironmentInstance != this)
	{
		// This is a duplicate instance - disable it
		bIsPrimaryEnvironment = false;
		bEnableTraining = false;  // Disable training on duplicate

		UE_LOG(LogTemp, Error, TEXT("╔════════════════════════════════════════════════════════════════╗"));
		UE_LOG(LogTemp, Error, TEXT("║ DUPLICATE ScholaCombatEnvironment DETECTED!                   ║"));
		UE_LOG(LogTemp, Error, TEXT("║ Primary: %s                                                   ║"), *PrimaryEnvironmentInstance->GetName());
		UE_LOG(LogTemp, Error, TEXT("║ Duplicate: %s (THIS - DISABLED)                               ║"), *GetName());
		UE_LOG(LogTemp, Error, TEXT("║ Training has been DISABLED on this instance.                  ║"));
		UE_LOG(LogTemp, Error, TEXT("║ ACTION: Remove duplicate ScholaCombatEnvironment from level   ║"));
		UE_LOG(LogTemp, Error, TEXT("╚════════════════════════════════════════════════════════════════╝"));

		// Still call parent to maintain proper lifecycle
		Super::BeginPlay();
		// But do NOT initialize this instance
		return;
	}

	// Check for multiple environment instances (common mistake)
	int32 EnvCount = 0;
	for (TActorIterator<AScholaCombatEnvironment> It(GetWorld()); It; ++It)
	{
		EnvCount++;
	}

	if (EnvCount > 1)
	{
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Found %d environments in level (only primary will be active)"), EnvCount);

		// List all environment instances
		int32 Index = 0;
		for (TActorIterator<AScholaCombatEnvironment> It(GetWorld()); It; ++It)
		{
			AScholaCombatEnvironment* Env = *It;
			UE_LOG(LogTemp, Warning, TEXT("  Environment #%d: %s (%s)"),
				Index++, *Env->GetName(),
				Env == PrimaryEnvironmentInstance ? TEXT("PRIMARY") : TEXT("DISABLED"));
		}
	}

	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] %s::BeginPlay() - Environment count: %d, Is Primary: %s"),
		*GetName(), EnvCount, bIsPrimaryEnvironment ? TEXT("YES") : TEXT("NO"));

	// Reset registration flag for new PIE session
	bAgentsRegistered = false;

	// Get SimulationManager
	SimulationManager = Cast<ASimulationManagerGameMode>(UGameplayStatics::GetGameMode(this));
	if (!SimulationManager)
	{
		UE_LOG(LogTemp, Error, TEXT("[ScholaEnv] SimulationManagerGameMode not found!"));
		return;
	}

	// Bind to episode events
	BindEpisodeEvents();

	// Auto-discover agents if enabled
	if (bAutoDiscoverAgents)
	{
		DiscoverAgents();
	}

	// NOW call Super::BeginPlay() to initialize Schola base class
	// (moved here to ensure our agents are discovered first)
	// NOTE: Super::BeginPlay() internally calls Initialize(), which triggers InternalRegisterAgents()
	// Do NOT call Initialize() manually again!
	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Calling Super::BeginPlay() (this will call Initialize() internally)..."));
	Super::BeginPlay();

	// Note: ScholaManagerSubsystem automatically handles server startup and agent registration.
	// We do not need to manually start the server here.

	// Debug: Check if environment is properly initialized
	if (GetWorld())
	{
		UGameInstance* GameInstance = GetWorld()->GetGameInstance();
		if (GameInstance)
		{
			UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] GameInstance: %s"), *GameInstance->GetClass()->GetName());
		}
	}

	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Initialized with %d agents (Training: %s, Port: %d)"),
		RegisteredAgents.Num(), bEnableTraining ? TEXT("ON") : TEXT("OFF"), ServerPort);
	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Base class: %s"), *GetClass()->GetSuperClass()->GetName());

	// List all registered agents
	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Registered agents:"));
	for (int32 i = 0; i < RegisteredAgents.Num(); i++)
	{
		UScholaAgentComponent* Agent = RegisteredAgents[i];
		if (Agent && Agent->GetOwner())
		{
			UE_LOG(LogTemp, Warning, TEXT("  [%d] %s (ActorName: %s)"),
				i, *Agent->GetName(), *Agent->GetOwner()->GetName());
		}
	}

	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Waiting for Python GymConnector to connect on port %d..."), ServerPort);
}

void AScholaCombatEnvironment::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	// Clear singleton reference if this was the primary instance
	if (PrimaryEnvironmentInstance == this)
	{
		PrimaryEnvironmentInstance = nullptr;
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Primary environment %s ending - clearing singleton"), *GetName());
	}

	Super::EndPlay(EndPlayReason);
}

//------------------------------------------------------------------------------
// SCHOLA ENVIRONMENT INTERFACE
//------------------------------------------------------------------------------

void AScholaCombatEnvironment::InitializeEnvironment()
{
	// Called by AAbstractScholaEnvironment::Initialize()
	// Setup any environment-specific initialization here
	UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] InitializeEnvironment called"));
}

void AScholaCombatEnvironment::ResetEnvironment()
{
	// Called by Schola when episode resets
	// The SimulationManager already handles agent reset, so we just log
	UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] ResetEnvironment called"));

	// Ensure simulation is running (starts it on first connection/reset)
	if (SimulationManager && !SimulationManager->IsSimulationRunning())
	{
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] First reset received - Starting Simulation"));
		SimulationManager->StartSimulation();
	}
}

void AScholaCombatEnvironment::InternalRegisterAgents(TArray<FTrainerAgentPair>& OutAgentTrainerPairs)
{
	// Called by AAbstractScholaEnvironment::Initialize()
	// Create AAbstractTrainer actors for each registered agent

	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] === InternalRegisterAgents CALLED ==="));
	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] RegisteredAgents.Num() = %d"), RegisteredAgents.Num());
	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] bAgentsRegistered = %s"), bAgentsRegistered ? TEXT("TRUE") : TEXT("FALSE"));

	// Guard against duplicate calls within same session
	if (bAgentsRegistered)
	{
		UE_LOG(LogTemp, Error, TEXT("[ScholaEnv] InternalRegisterAgents already called this session! Skipping duplicate registration."));
		UE_LOG(LogTemp, Error, TEXT("[ScholaEnv] This may indicate a Schola initialization bug or multiple environment instances."));
		return;
	}

	OutAgentTrainerPairs.Empty();

	// WORKAROUND: Schola's plugin discovers ALL components (including CDO) for action_space,
	// but we only registered non-CDO agents. This creates action_space/id_manager mismatch.
	// Solution: Find all components (including CDO) and create trainers for all of them.
	// CDO trainers will be inactive but present in id_manager to prevent index errors.

	TArray<UScholaAgentComponent*> AllComponents;

	// Find ALL ScholaAgentComponents (including CDO) using global iterator
	// This matches what Schola's plugin does when creating action_space
	for (TObjectIterator<UScholaAgentComponent> It; It; ++It)
	{
		UScholaAgentComponent* Comp = *It;
		// Only include components in this world (exclude other PIE instances, etc.)
		// BUT: We MUST include CDOs because Schola discovers them and expects trainers for them
		bool bIsCDO = Comp->HasAnyFlags(RF_ClassDefaultObject | RF_ArchetypeObject);
		bool bInThisWorld = (Comp->GetWorld() == GetWorld());

		// Include if it's in this world OR if it's a CDO (which usually has no world)
		// Note: We assume CDOs don't belong to *other* worlds. If they do, we might need more checks.
		if (bInThisWorld || bIsCDO)
		{
			AllComponents.Add(Comp);

			FString OwnerName = Comp->GetOwner() ? Comp->GetOwner()->GetName() : TEXT("NO_OWNER");
			bool bIsRegistered = RegisteredAgents.Contains(Comp);

			UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Found component: %s | Owner: %s | IsCDO: %s | IsRegistered: %s"),
				*Comp->GetName(),
				*OwnerName,
				bIsCDO ? TEXT("YES") : TEXT("NO"),
				bIsRegistered ? TEXT("YES") : TEXT("NO"));
		}
	}

	// Also check for CDOs outside this world (might be in blueprint package)
	for (TObjectIterator<UScholaAgentComponent> It; It; ++It)
	{
		UScholaAgentComponent* Comp = *It;
		if (Comp && Comp->HasAnyFlags(RF_ClassDefaultObject | RF_ArchetypeObject))
		{
			UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Found CDO in other context: %s (World: %s)"),
				*Comp->GetName(),
				Comp->GetWorld() ? *Comp->GetWorld()->GetName() : TEXT("NULL"));
		}
	}

	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Total components found (including CDOs): %d"), AllComponents.Num());
	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Registered agents (non-CDO): %d"), RegisteredAgents.Num());

	int32 TrainersCreated = 0;
	int32 TrainersFailed = 0;
	int32 DummyTrainersCreated = 0;

	// Process all components in the order Schola discovers them
	// This ensures id_manager indices match action_space keys
	for (int32 i = 0; i < AllComponents.Num(); i++)
	{
		UScholaAgentComponent* Agent = AllComponents[i];
		if (!Agent)
		{
			UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] AllComponents[%d] is null, skipping"), i);
			TrainersFailed++;
			continue;
		}

		// Check if this is a CDO (Class Default Object)
		bool bIsCDO = Agent->HasAnyFlags(RF_ClassDefaultObject | RF_ArchetypeObject);
		bool bHasValidOwner = Agent->GetOwner() && !Agent->GetOwner()->HasAnyFlags(RF_ClassDefaultObject);

		if (bIsCDO || !bHasValidOwner)
		{
			// CDO or invalid component - create a DUMMY trainer to occupy id_manager slot
			UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Processing CDO/Invalid component %d/%d: %s (creating dummy trainer)"),
				i + 1, AllComponents.Num(), *Agent->GetName());

			// Spawn a dummy pawn for the trainer to possess (required by AbstractTrainer::Initialize)
			FActorSpawnParameters PawnParams;
			PawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
			
			// Use ADefaultPawn as a safe, lightweight dummy
			APawn* DummyPawn = GetWorld()->SpawnActor<APawn>(
				ADefaultPawn::StaticClass(), // Use concrete ADefaultPawn
				FVector(0, 0, -10000), // Spawn far away
				FRotator::ZeroRotator,
				PawnParams
			);

			// Create minimal dummy trainer
			FActorSpawnParameters TrainerParams;
			TrainerParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

			AFollowerAgentTrainer* DummyTrainer = GetWorld()->SpawnActor<AFollowerAgentTrainer>(
				AFollowerAgentTrainer::StaticClass(),
				FVector::ZeroVector,
				FRotator::ZeroRotator,
				TrainerParams
			);

			if (DummyTrainer && DummyPawn)
			{
				// Possess the dummy pawn (required for GetPawn() calls in Initialize)
				DummyTrainer->Possess(DummyPawn);

				// Add to output array with the dummy pawn
				FTrainerAgentPair Pair(DummyPawn, DummyTrainer);
				OutAgentTrainerPairs.Add(Pair);

				DummyTrainersCreated++;
				UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv]   - ✓ Created DUMMY trainer: %s (possessing %s)"), 
					*DummyTrainer->GetName(), *DummyPawn->GetName());
			}
			else
			{
				TrainersFailed++;
				UE_LOG(LogTemp, Error, TEXT("[ScholaEnv]   - ✗ Failed to spawn dummy trainer/pawn for %s"), *Agent->GetName());
				
				if (DummyPawn) DummyPawn->Destroy();
				if (DummyTrainer) DummyTrainer->Destroy();
			}

			continue;
		}

		// Real agent - process normally
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Processing agent %d/%d: %s"),
			i + 1, AllComponents.Num(), *Agent->GetOwner()->GetName());

		// Initialize agent if not already done (fixes timing issue)
		if (!Agent->FollowerAgent)
		{
			UE_LOG(LogTemp, Log, TEXT("[ScholaEnv]   - FollowerAgent null, calling InitializeScholaComponents()"));
			Agent->InitializeScholaComponents();
		}

		// Validate after initialization
		if (!Agent->FollowerAgent)
		{
			UE_LOG(LogTemp, Error, TEXT("[ScholaEnv]   - ✗ Agent %s has no FollowerAgent after initialization!"),
				*Agent->GetOwner()->GetName());
			TrainersFailed++;
			continue;
		}

		UE_LOG(LogTemp, Log, TEXT("[ScholaEnv]   - ✓ FollowerAgent validated"));

		// Spawn FollowerAgentTrainer actor (controller owns pawn, not vice versa)
		FActorSpawnParameters SpawnParams;
		SpawnParams.NameMode = FActorSpawnParameters::ESpawnActorNameMode::Requested;
		SpawnParams.Name = FName(*FString::Printf(TEXT("Trainer_%s"), *Agent->GetOwner()->GetName()));

		AFollowerAgentTrainer* Trainer = GetWorld()->SpawnActor<AFollowerAgentTrainer>(
			AFollowerAgentTrainer::StaticClass(),
			Agent->GetOwner()->GetActorLocation(),
			FRotator::ZeroRotator,
			SpawnParams
		);

		if (Trainer)
		{
			// Link trainer to agent
			Trainer->Initialize(Agent);

			// Add to output array (required by Schola)
			FTrainerAgentPair Pair(Agent->GetControlledPawn(), Trainer);
			OutAgentTrainerPairs.Add(Pair);

			TrainersCreated++;
			UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv]   - ✓ Created trainer: %s"), *Trainer->GetName());
		}
		else
		{
			TrainersFailed++;
			UE_LOG(LogTemp, Error, TEXT("[ScholaEnv]   - ✗ Failed to spawn trainer for %s"), *Agent->GetOwner()->GetName());
		}
	}

	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] === TRAINER REGISTRATION COMPLETE ==="));
	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Real Trainers: %d | Dummy Trainers: %d | Failed: %d | Total Pairs: %d"),
		TrainersCreated, DummyTrainersCreated, TrainersFailed, OutAgentTrainerPairs.Num());

	// List all trainer-agent pairs
	for (int32 i = 0; i < OutAgentTrainerPairs.Num(); i++)
	{
		const FTrainerAgentPair& Pair = OutAgentTrainerPairs[i];
		if (Pair.AgentCDO && Pair.Trainer)
		{
			UE_LOG(LogTemp, Warning, TEXT("  [%d] Agent=%s, Trainer=%s"),
				i, *Pair.AgentCDO->GetName(), *Pair.Trainer->GetName());
		}
		else
		{
			UE_LOG(LogTemp, Error, TEXT("  [%d] INVALID PAIR (Agent=%s, Trainer=%s)"),
				i, Pair.AgentCDO ? *Pair.AgentCDO->GetName() : TEXT("NULL"),
				Pair.Trainer ? *Pair.Trainer->GetName() : TEXT("NULL"));
		}
	}

	// Mark as registered to prevent duplicates within this session
	bAgentsRegistered = true;
	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] bAgentsRegistered set to TRUE"));
}

void AScholaCombatEnvironment::SetEnvironmentOptions(const TMap<FString, FString>& Options)
{
	// Called by Schola GymConnector to configure environment
	// Can be used to pass settings from Python (e.g., difficulty, map variant)
	UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] SetEnvironmentOptions called with %d options"), Options.Num());

	for (const auto& Pair : Options)
	{
		UE_LOG(LogTemp, Log, TEXT("  %s = %s"), *Pair.Key, *Pair.Value);
	}
}

void AScholaCombatEnvironment::SeedEnvironment(int Seed)
{
	// Called by Schola GymConnector to seed randomness
	UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] SeedEnvironment called with seed %d"), Seed);
	FMath::RandInit(Seed);
}

//------------------------------------------------------------------------------
// AGENT DISCOVERY & REGISTRATION
//------------------------------------------------------------------------------

void AScholaCombatEnvironment::DiscoverAgents()
{
	RegisteredAgents.Empty();

	int32 TotalFound = 0;
	int32 ValidatedCount = 0;
	int32 SkippedCount = 0;

	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] === AGENT DISCOVERY START ==="));

	// Find all ScholaAgentComponents in level
	for (TActorIterator<AActor> It(GetWorld()); It; ++It)
	{
		AActor* Actor = *It;
		if (!Actor)
		{
			continue;
		}

		UScholaAgentComponent* ScholaComp = Actor->FindComponentByClass<UScholaAgentComponent>();
		if (ScholaComp)
		{
			// CRITICAL: Filter out CDO (Class Default Object)
			// CDOs are template objects and should never be registered for training
			if (ScholaComp->HasAnyFlags(RF_ClassDefaultObject) || ScholaComp->HasAnyFlags(RF_ArchetypeObject))
			{
				UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Skipping CDO/Archetype: %s"), *ScholaComp->GetName());
				continue;
			}

			// Also check if the component's owner is valid and not a CDO
			if (!Actor || Actor->HasAnyFlags(RF_ClassDefaultObject))
			{
				UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Skipping component with CDO owner: %s"), *ScholaComp->GetName());
				continue;
			}

			TotalFound++;

			// Log detailed info about discovered agent
			int32 TeamID = SimulationManager ? SimulationManager->GetTeamIDForActor(Actor) : -1;
			UFollowerAgentComponent* FollowerComp = Actor->FindComponentByClass<UFollowerAgentComponent>();
			UTeamLeaderComponent* LeaderComp = Actor->FindComponentByClass<UTeamLeaderComponent>();

			UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Found Agent #%d:"), TotalFound);
			UE_LOG(LogTemp, Warning, TEXT("  - Component Name: %s"), *ScholaComp->GetName());
			UE_LOG(LogTemp, Warning, TEXT("  - Owner Actor: %s"), *Actor->GetName());
			UE_LOG(LogTemp, Warning, TEXT("  - Owner Class: %s"), *Actor->GetClass()->GetName());
			UE_LOG(LogTemp, Warning, TEXT("  - Team ID: %d"), TeamID);
			UE_LOG(LogTemp, Warning, TEXT("  - Has FollowerComponent: %s"), FollowerComp ? TEXT("YES") : TEXT("NO"));
			UE_LOG(LogTemp, Warning, TEXT("  - Has LeaderComponent: %s"), LeaderComp ? TEXT("YES") : TEXT("NO"));
			UE_LOG(LogTemp, Warning, TEXT("  - Actor Pending Kill: %s"), Actor->IsPendingKillPending() ? TEXT("YES") : TEXT("NO"));
			UE_LOG(LogTemp, Warning, TEXT("  - Actor Hidden: %s"), Actor->IsHidden() ? TEXT("YES") : TEXT("NO"));

			// CRITICAL: Skip team leaders (leaders should NOT have ScholaAgentComponent!)
			if (LeaderComp)
			{
				SkippedCount++;
				UE_LOG(LogTemp, Error, TEXT("  - Status: SKIPPED - TEAM LEADER MISCONFIGURED (has ScholaAgentComponent!)"));
				UE_LOG(LogTemp, Error, TEXT("  - ACTION REQUIRED: Remove ScholaAgentComponent from leader %s"), *Actor->GetName());
				continue;
			}

			bool bRegistered = RegisterAgent(ScholaComp);
			if (bRegistered)
			{
				ValidatedCount++;
				UE_LOG(LogTemp, Warning, TEXT("  - Status: REGISTERED"));
			}
			else
			{
				SkippedCount++;
				UE_LOG(LogTemp, Warning, TEXT("  - Status: SKIPPED (validation failed)"));
			}
		}
	}

	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] === AGENT DISCOVERY COMPLETE ==="));
	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Total Found: %d | Registered: %d | Skipped: %d"),
		TotalFound, ValidatedCount, SkippedCount);
	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Final RegisteredAgents.Num(): %d"), RegisteredAgents.Num());
}

bool AScholaCombatEnvironment::RegisterAgent(UScholaAgentComponent* Agent)
{
	if (!Agent)
	{
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] RegisterAgent failed: Agent is null"));
		return false;
	}

	// Validate agent has required components
	if (!ValidateAgent(Agent))
	{
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Agent %s failed validation"), *Agent->GetOwner()->GetName());
		return false;
	}

	// Check team filter
	if (TrainingTeamIDs.Num() > 0 && SimulationManager)
	{
		int32 TeamID = SimulationManager->GetTeamIDForActor(Agent->GetOwner());
		if (!TrainingTeamIDs.Contains(TeamID))
		{
			UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Skipping agent %s (Team %d not in training list: [%s])"),
				*Agent->GetOwner()->GetName(), TeamID, *FString::JoinBy(TrainingTeamIDs, TEXT(", "),
				[](int32 ID) { return FString::FromInt(ID); }));
			return false;
		}
		else
		{
			UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] Team %d is in training list"), TeamID);
		}
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] No team filter active (TrainingTeamIDs is empty)"));
	}

	// Add to registered agents
	RegisteredAgents.AddUnique(Agent);

	// Link agent to this environment
	Agent->ScholaEnvironment = this;

	UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] ✓ Successfully registered agent: %s"), *Agent->GetOwner()->GetName());
	return true;
}

bool AScholaCombatEnvironment::ValidateAgent(UScholaAgentComponent* Agent) const
{
	if (!Agent || !Agent->GetOwner())
	{
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Validation failed: Agent or Owner is null"));
		return false;
	}

	FString ActorName = Agent->GetOwner()->GetName();
	bool bIsValid = true;

	// Check for required FollowerAgentComponent
	UFollowerAgentComponent* FollowerComp = Agent->GetOwner()->FindComponentByClass<UFollowerAgentComponent>();
	if (!FollowerComp)
	{
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] ✗ %s: Missing FollowerAgentComponent"), *ActorName);
		bIsValid = false;
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] ✓ %s: Has FollowerAgentComponent"), *ActorName);
	}

	// Check for TacticalObserver
	if (!Agent->TacticalObserver)
	{
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] ✗ %s: Missing TacticalObserver"), *ActorName);
		bIsValid = false;
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] ✓ %s: Has TacticalObserver"), *ActorName);
	}

	// Check for RewardProvider
	if (!Agent->RewardProvider)
	{
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] ✗ %s: Missing RewardProvider"), *ActorName);
		bIsValid = false;
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] ✓ %s: Has RewardProvider"), *ActorName);
	}

	// Check for TacticalActuator
	if (!Agent->TacticalActuator)
	{
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] ✗ %s: Missing TacticalActuator"), *ActorName);
		bIsValid = false;
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] ✓ %s: Has TacticalActuator"), *ActorName);
	}

	if (bIsValid)
	{
		UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] ✓ %s: Validation PASSED"), *ActorName);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] ✗ %s: Validation FAILED"), *ActorName);
	}

	return bIsValid;
}

//------------------------------------------------------------------------------
// TRAINING SERVER
//------------------------------------------------------------------------------

bool AScholaCombatEnvironment::StartTrainingServer()
{
	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] StartTrainingServer is deprecated. ScholaManagerSubsystem handles server startup automatically."));
	return true;
}

void AScholaCombatEnvironment::StopTrainingServer()
{
	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] StopTrainingServer is deprecated. ScholaManagerSubsystem handles server shutdown automatically."));
}

//------------------------------------------------------------------------------
// EPISODE EVENTS
//------------------------------------------------------------------------------

void AScholaCombatEnvironment::BindEpisodeEvents()
{
	if (!SimulationManager)
	{
		return;
	}

	// Bind to episode lifecycle events
	SimulationManager->OnEpisodeStarted.AddUniqueDynamic(this, &AScholaCombatEnvironment::OnEpisodeStarted);
	SimulationManager->OnEpisodeEnded.AddUniqueDynamic(this, &AScholaCombatEnvironment::OnEpisodeEnded);

	UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] Bound to episode events"));
}

void AScholaCombatEnvironment::OnEpisodeStarted(int32 EpisodeNumber)
{
	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Episode %d started - Resetting agents"), EpisodeNumber);

	// Reset all registered agents
	for (UScholaAgentComponent* Agent : RegisteredAgents)
	{
		if (Agent)
		{
			Agent->ResetEpisode();
		}
	}

	// Notify Schola environment (calls ResetEnvironment)
	Reset();
}

void AScholaCombatEnvironment::OnEpisodeEnded(const FEpisodeResult& Result)
{
	UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Episode %d ended - Winner: Team %d, Duration: %.2fs, Steps: %d"),
		Result.EpisodeNumber, Result.WinningTeamID, Result.EpisodeDuration, Result.TotalSteps);

	// Mark environment as completed (triggers Schola to collect final experiences)
	MarkCompleted();
}
