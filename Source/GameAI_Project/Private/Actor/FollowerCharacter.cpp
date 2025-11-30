// Copyright Epic Games, Inc. All Rights Reserved.

#include "Actor/FollowerCharacter.h"
#include "Team/FollowerAgentComponent.h"
#include "StateTree/FollowerStateTreeComponent.h"
#include "Schola/ScholaAgentComponent.h"
#include "Combat/HealthComponent.h"
#include "Combat/WeaponComponent.h"
#include "GameFramework/CharacterMovementComponent.h"

AFollowerCharacter::AFollowerCharacter()
{
	PrimaryActorTick.bCanEverTick = true;

	// Create follower agent component
	FollowerAgentComponent = CreateDefaultSubobject<UFollowerAgentComponent>(TEXT("FollowerAgentComponent"));

	// Create state tree component
	StateTreeComponent = CreateDefaultSubobject<UFollowerStateTreeComponent>(TEXT("StateTreeComponent"));

	// Create Schola agent component (RLlib training bridge)
	ScholaAgentComponent = CreateDefaultSubobject<UScholaAgentComponent>(TEXT("ScholaAgentComponent"));

	// Configure character movement for AI pathfinding
	if (UCharacterMovementComponent* MoveComp = GetCharacterMovement())
	{
		MoveComp->bOrientRotationToMovement = true;
		MoveComp->RotationRate = FRotator(0.0f, 540.0f, 0.0f);
		MoveComp->MaxWalkSpeed = 600.0f;
		MoveComp->bUseControllerDesiredRotation = false;
	}

	// Don't rotate character based on controller - let movement component handle it
	bUseControllerRotationPitch = false;
	bUseControllerRotationYaw = false;
	bUseControllerRotationRoll = false;
}

void AFollowerCharacter::BeginPlay()
{
	Super::BeginPlay();

	UE_LOG(LogTemp, Warning, TEXT("AFollowerCharacter::BeginPlay: '%s' - StateTreeComponent=%s, FollowerAgentComponent=%s, ScholaAgentComponent=%s"),
		*GetName(),
		StateTreeComponent ? TEXT("✅ Valid") : TEXT("❌ NULL"),
		FollowerAgentComponent ? TEXT("✅ Valid") : TEXT("❌ NULL"),
		ScholaAgentComponent ? TEXT("✅ Valid") : TEXT("❌ NULL"));
}

void AFollowerCharacter::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

//------------------------------------------------------------------------------
// COMBAT STATS (ICombatStatsInterface Implementation)
// Delegates to HealthComponent and WeaponComponent
//------------------------------------------------------------------------------

float AFollowerCharacter::GetHealthPercentage_Implementation() const
{
	UHealthComponent* HealthComp = FindComponentByClass<UHealthComponent>();
	return HealthComp ? HealthComp->GetHealthPercentage() * 100.0f : 0.0f;
}

bool AFollowerCharacter::IsAlive_Implementation() const
{
	UHealthComponent* HealthComp = FindComponentByClass<UHealthComponent>();
	return HealthComp ? HealthComp->IsAlive() : false;
}

float AFollowerCharacter::GetWeaponCooldown_Implementation() const
{
	UWeaponComponent* WeaponComp = FindComponentByClass<UWeaponComponent>();
	return WeaponComp ? WeaponComp->GetRemainingCooldown() : 0.0f;
}

bool AFollowerCharacter::CanFireWeapon_Implementation() const
{
	UWeaponComponent* WeaponComp = FindComponentByClass<UWeaponComponent>();
	return WeaponComp ? WeaponComp->CanFire() : false;
}
