// Copyright Epic Games, Inc. All Rights Reserved.

#include "Actor/FollowerCharacter.h"
#include "Team/FollowerAgentComponent.h"
#include "StateTree/FollowerStateTreeComponent.h"
#include "GameFramework/CharacterMovementComponent.h"

AFollowerCharacter::AFollowerCharacter()
{
	PrimaryActorTick.bCanEverTick = true;

	// Create follower agent component
	FollowerAgentComponent = CreateDefaultSubobject<UFollowerAgentComponent>(TEXT("FollowerAgentComponent"));

	// Create state tree component
	StateTreeComponent = CreateDefaultSubobject<UFollowerStateTreeComponent>(TEXT("StateTreeComponent"));

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

	UE_LOG(LogTemp, Warning, TEXT("AFollowerCharacter::BeginPlay: '%s' - StateTreeComponent=%s, FollowerAgentComponent=%s"),
		*GetName(),
		StateTreeComponent ? TEXT("✅ Valid") : TEXT("❌ NULL"),
		FollowerAgentComponent ? TEXT("✅ Valid") : TEXT("❌ NULL"));
}

void AFollowerCharacter::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	// Update combat timers
	UpdateTimers(DeltaTime);
}

//------------------------------------------------------------------------------
// COMBAT STATS (ICombatStatsInterface Implementation)
//------------------------------------------------------------------------------

float AFollowerCharacter::GetHealthPercentage_Implementation() const
{
	return (CombatStats.MaxHealth > 0.0f) ? (CombatStats.CurrentHealth / CombatStats.MaxHealth) * 100.0f : 0.0f;
}

bool AFollowerCharacter::IsAlive_Implementation() const
{
	return !bIsDead && CombatStats.CurrentHealth > 0.0f;
}

float AFollowerCharacter::GetWeaponCooldown_Implementation() const
{
	return CombatStats.CurrentWeaponCooldown;
}


bool AFollowerCharacter::CanFireWeapon_Implementation() const
{
	return IsAlive_Implementation() && CombatStats.CurrentWeaponCooldown <= 0.0f;
}

//------------------------------------------------------------------------------
// HEALTH SYSTEM
//------------------------------------------------------------------------------

void AFollowerCharacter::TakeDamage(float DamageAmount)
{
	if (bIsDead || DamageAmount <= 0.0f)
	{
		return;
	}


	// Apply remaining to health
	if (DamageAmount > 0.0f)
	{
		CombatStats.CurrentHealth = FMath::Max(0.0f, CombatStats.CurrentHealth - DamageAmount);
	}

	// Reset shield regen timer
	TimeSinceLastDamage = 0.0f;

	// Check for death
	if (CombatStats.CurrentHealth <= 0.0f)
	{
		Kill();
	}
}

void AFollowerCharacter::Heal(float HealAmount)
{
	if (bIsDead)
	{
		return;
	}

	CombatStats.CurrentHealth = FMath::Min(CombatStats.MaxHealth, CombatStats.CurrentHealth + HealAmount);
}

void AFollowerCharacter::Kill()
{
	if (bIsDead)
	{
		return;
	}

	bIsDead = true;
	CombatStats.CurrentHealth = 0.0f;

	// Notify follower component
	if (FollowerAgentComponent)
	{
		FollowerAgentComponent->MarkAsDead();
	}

	UE_LOG(LogTemp, Log, TEXT("FollowerCharacter '%s' died"), *GetName());
}

void AFollowerCharacter::Respawn()
{
	bIsDead = false;
	CombatStats.CurrentHealth = CombatStats.MaxHealth;
	CombatStats.CurrentWeaponCooldown = 0.0f;

	// Notify follower component
	if (FollowerAgentComponent)
	{
		FollowerAgentComponent->MarkAsAlive();
		FollowerAgentComponent->ResetEpisode();
	}

	UE_LOG(LogTemp, Log, TEXT("FollowerCharacter '%s' respawned"), *GetName());
}


//------------------------------------------------------------------------------
// WEAPON SYSTEM
//------------------------------------------------------------------------------

bool AFollowerCharacter::FireWeapon()
{
	if (!CanFireWeapon_Implementation())
	{
		return false;
	}


	// Start cooldown
	CombatStats.CurrentWeaponCooldown = CombatStats.CooldownTime;

	return true;
}


//------------------------------------------------------------------------------
// PRIVATE
//------------------------------------------------------------------------------

void AFollowerCharacter::UpdateTimers(float DeltaTime)
{
	UpdateWeaponCooldown(DeltaTime);
}



void AFollowerCharacter::UpdateWeaponCooldown(float DeltaTime)
{
	if (CombatStats.CurrentWeaponCooldown > 0.0f)
	{
		CombatStats.CurrentWeaponCooldown = FMath::Max(0.0f, CombatStats.CurrentWeaponCooldown - DeltaTime);
	}
}
