// ScholaAgentComponent.cpp - Schola agent component implementation

#include "Schola/ScholaAgentComponent.h"
#include "Schola/TacticalObserver.h"
#include "Schola/TacticalRewardProvider.h"
#include "Schola/TacticalActuator.h"
#include "Team/FollowerAgentComponent.h"
#include "Inference/InferenceComponent.h"
#include "GameFramework/Pawn.h"

UScholaAgentComponent::UScholaAgentComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.bStartWithTickEnabled = true;

	// Create default subobjects for Schola components
	TacticalObserver = CreateDefaultSubobject<UTacticalObserver>(TEXT("TacticalObserver"));
	RewardProvider = CreateDefaultSubobject<UTacticalRewardProvider>(TEXT("RewardProvider"));
	TacticalActuator = CreateDefaultSubobject<UTacticalActuator>(TEXT("TacticalActuator"));
}

void UScholaAgentComponent::BeginPlay()
{
	Super::BeginPlay();

	// Auto-configure follower agent if enabled
	if (bAutoConfigureFollower)
	{
		InitializeScholaComponents();
	}

	UE_LOG(LogTemp, Log, TEXT("[ScholaAgent] %s: Initialized (Training=%s, Port=%d)"),
		*GetOwner()->GetName(),
		bEnableScholaTraining ? TEXT("ON") : TEXT("OFF"),
		ScholaServerPort);
}

void UScholaAgentComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);

	FollowerAgent = nullptr;
}

void UScholaAgentComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// This component mainly serves as a bridge/configuration helper
	// The actual Think/Act cycle is handled by Schola's InferenceComponent
}

void UScholaAgentComponent::InitializeScholaComponents()
{
	// Find follower agent component
	FollowerAgent = FindFollowerAgent();
	if (!FollowerAgent)
	{
		UE_LOG(LogTemp, Warning, TEXT("[ScholaAgent] %s: FollowerAgentComponent not found!"),
			*GetOwner()->GetName());
		return;
	}

	// Configure components
	ConfigureObservers();
	ConfigureRewardProvider();
	ConfigureActuators();

	UE_LOG(LogTemp, Log, TEXT("[ScholaAgent] %s: Schola components configured successfully"),
		*GetOwner()->GetName());
}

void UScholaAgentComponent::ConfigureObservers()
{
	if (!TacticalObserver || !FollowerAgent)
	{
		return;
	}

	// Link observer to follower agent
	TacticalObserver->FollowerAgent = FollowerAgent;
	TacticalObserver->bAutoFindFollower = false;
	TacticalObserver->InitializeObserver();

	// Add to InferenceComponent's observers array if not already present (this class IS the InferenceComponent)
	if (!this->Observers.Contains(TacticalObserver))
	{
		this->Observers.Add(TacticalObserver);
	}

	UE_LOG(LogTemp, Verbose, TEXT("[ScholaAgent] %s: TacticalObserver configured (71 features)"),
		*GetOwner()->GetName());
}

void UScholaAgentComponent::ConfigureRewardProvider()
{
	if (!RewardProvider || !FollowerAgent)
	{
		return;
	}

	// Link reward provider to follower agent
	RewardProvider->FollowerAgent = FollowerAgent;
	RewardProvider->bAutoFindFollower = false;
	RewardProvider->Initialize();

	UE_LOG(LogTemp, Verbose, TEXT("[ScholaAgent] %s: RewardProvider configured"),
		*GetOwner()->GetName());
}

UFollowerAgentComponent* UScholaAgentComponent::FindFollowerAgent() const
{
	AActor* Owner = GetOwner();
	if (!Owner)
	{
		return nullptr;
	}

	return Owner->FindComponentByClass<UFollowerAgentComponent>();
}

float UScholaAgentComponent::GetCurrentReward() const
{
	if (!RewardProvider)
	{
		return 0.0f;
	}

	return RewardProvider->GetReward();
}

bool UScholaAgentComponent::IsEpisodeTerminated() const
{
	if (!RewardProvider)
	{
		return false;
	}

	return RewardProvider->IsTerminated();
}

void UScholaAgentComponent::ConfigureActuators()
{
	if (!TacticalActuator || !FollowerAgent)
	{
		return;
	}

	// Link actuator to follower agent
	TacticalActuator->FollowerAgent = FollowerAgent;
	TacticalActuator->bAutoFindFollower = false;
	TacticalActuator->InitializeActuator();

	// Add to InferenceComponent's actuators array if not already present (this class IS the InferenceComponent)
	if (!this->Actuators.Contains(TacticalActuator))
	{
		this->Actuators.Add(TacticalActuator);
	}

	UE_LOG(LogTemp, Verbose, TEXT("[ScholaAgent] %s: TacticalActuator configured (8D actions)"),
		*GetOwner()->GetName());
}

void UScholaAgentComponent::ResetEpisode()
{
	// Reset reward provider
	if (RewardProvider)
	{
		RewardProvider->Reset();
	}

	// Reset observer
	if (TacticalObserver)
	{
		TacticalObserver->ResetObserver();
	}

	// Reset follower agent episode
	if (FollowerAgent)
	{
		FollowerAgent->ResetEpisode();
	}

	UE_LOG(LogTemp, Verbose, TEXT("[ScholaAgent] %s: Episode reset"),
		*GetOwner()->GetName());
}
