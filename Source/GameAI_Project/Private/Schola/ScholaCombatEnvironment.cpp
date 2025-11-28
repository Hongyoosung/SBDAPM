// ScholaCombatEnvironment.cpp - Schola environment implementation

#include "Schola/ScholaCombatEnvironment.h"
#include "Schola/ScholaAgentComponent.h"
#include "Schola/FollowerAgentTrainer.h"
#include "Core/SimulationManagerGameMode.h"
#include "Core/ScholaGameInstance.h"
#include "Team/FollowerAgentComponent.h"
#include "Communicator/CommunicationManager.h"
#include "Subsystem/ScholaManagerSubsystem.h"
#include "EngineUtils.h"
#include "Kismet/GameplayStatics.h"

AScholaCombatEnvironment::AScholaCombatEnvironment(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	PrimaryActorTick.bCanEverTick = false;
}

void AScholaCombatEnvironment::BeginPlay()
{
	Super::BeginPlay();

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

	// Initialize environment (calls parent AAbstractScholaEnvironment::Initialize)
	Initialize();

	// Note: ScholaManagerSubsystem automatically handles server startup and agent registration.
	// We do not need to manually start the server here.

	UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] Initialized with %d agents (Training: %s, Port: %d)"),
		RegisteredAgents.Num(), bEnableTraining ? TEXT("ON") : TEXT("OFF"), ServerPort);
}

void AScholaCombatEnvironment::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
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
}

void AScholaCombatEnvironment::InternalRegisterAgents(TArray<FTrainerAgentPair>& OutAgentTrainerPairs)
{
	// Called by AAbstractScholaEnvironment::Initialize()
	// Create AAbstractTrainer actors for each registered agent

	OutAgentTrainerPairs.Empty();

	for (UScholaAgentComponent* Agent : RegisteredAgents)
	{
		if (!Agent)
		{
			continue;
		}

		// Initialize agent if not already done (fixes timing issue)
		if (!Agent->FollowerAgent)
		{
			Agent->InitializeScholaComponents();
		}

		// Validate after initialization
		if (!Agent->FollowerAgent)
		{
			UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Agent %s has no FollowerAgent after initialization"),
				*Agent->GetOwner()->GetName());
			continue;
		}

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

			UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] Created trainer for %s"), *Agent->GetOwner()->GetName());
		}
	}

	UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] Registered %d trainers"), OutAgentTrainerPairs.Num());
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
			RegisterAgent(ScholaComp);
		}
	}

	UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] Discovered %d agents"), RegisteredAgents.Num());
}

bool AScholaCombatEnvironment::RegisterAgent(UScholaAgentComponent* Agent)
{
	if (!Agent)
	{
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
			UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] Skipping agent %s (Team %d not in training list)"),
				*Agent->GetOwner()->GetName(), TeamID);
			return false;
		}
	}

	// Add to registered agents
	RegisteredAgents.AddUnique(Agent);

	// Link agent to this environment
	Agent->ScholaEnvironment = this;

	UE_LOG(LogTemp, Log, TEXT("[ScholaEnv] Registered agent: %s"), *Agent->GetOwner()->GetName());
	return true;
}

bool AScholaCombatEnvironment::ValidateAgent(UScholaAgentComponent* Agent) const
{
	if (!Agent || !Agent->GetOwner())
	{
		return false;
	}

	// Check for required FollowerAgentComponent
	UFollowerAgentComponent* FollowerComp = Agent->GetOwner()->FindComponentByClass<UFollowerAgentComponent>();
	if (!FollowerComp)
	{
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Agent %s missing FollowerAgentComponent"),
			*Agent->GetOwner()->GetName());
		return false;
	}

	// Check for TacticalObserver, RewardProvider, TacticalActuator
	if (!Agent->TacticalObserver)
	{
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Agent %s missing TacticalObserver"),
			*Agent->GetOwner()->GetName());
		return false;
	}

	if (!Agent->RewardProvider)
	{
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Agent %s missing RewardProvider"),
			*Agent->GetOwner()->GetName());
		return false;
	}

	if (!Agent->TacticalActuator)
	{
		UE_LOG(LogTemp, Warning, TEXT("[ScholaEnv] Agent %s missing TacticalActuator"),
			*Agent->GetOwner()->GetName());
		return false;
	}

	return true;
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
