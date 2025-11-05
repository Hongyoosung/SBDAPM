// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "Interfaces/CombatStatsInterface.h"
#include "FollowerCharacter.generated.h"

class UFollowerAgentComponent;
class UFollowerStateTreeComponent;
class URLPolicyNetwork;

/**
 * Follower Character - State Tree Based
 *
 * This character integrates the hierarchical multi-agent system using State Tree.
 *
 * Architecture:
 * - FollowerAgentComponent: Receives commands from team leader, manages RL policy
 * - FollowerStateTreeComponent: Executes tactical states (Assault, Defend, Support)
 * - Combat stats: Health, stamina, weapon system
 *
 * Usage:
 * 1. Spawn this character with FollowerAIController
 * 2. Set TeamLeaderActor in FollowerAgentComponent
 * 3. Assign State Tree asset in FollowerStateTreeComponent
 * 4. System auto-starts on BeginPlay
 */
UCLASS()
class GAMEAI_PROJECT_API AFollowerCharacter : public ACharacter, public ICombatStatsInterface
{
	GENERATED_BODY()

public:
	AFollowerCharacter();

protected:
	virtual void BeginPlay() override;

public:
	virtual void Tick(float DeltaTime) override;

	//--------------------------------------------------------------------------
	// COMPONENTS
	//--------------------------------------------------------------------------

	/** Follower agent component (command execution, RL policy) */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "AI|Components")
	UFollowerAgentComponent* FollowerAgentComponent;

	/** State Tree component (tactical state management) */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "AI|Components")
	UFollowerStateTreeComponent* StateTreeComponent;

	//--------------------------------------------------------------------------
	// COMBAT STATS (ICombatStatsInterface Implementation)
	//--------------------------------------------------------------------------

	virtual float GetHealthPercentage_Implementation() const override;
	virtual float GetStaminaPercentage_Implementation() const override;
	virtual float GetShieldPercentage_Implementation() const override;
	virtual bool IsAlive_Implementation() const override;
	virtual float GetWeaponCooldown_Implementation() const override;
	virtual float GetAmmunition_Implementation() const override;
	virtual int32 GetWeaponType_Implementation() const override;
	virtual bool CanFireWeapon_Implementation() const override;

	//--------------------------------------------------------------------------
	// HEALTH SYSTEM
	//--------------------------------------------------------------------------

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Health")
	float MaxHealth = 100.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Health")
	float CurrentHealth = 100.0f;

	UFUNCTION(BlueprintCallable, Category = "Combat|Health")
	void TakeDamage(float DamageAmount);

	UFUNCTION(BlueprintCallable, Category = "Combat|Health")
	void Heal(float HealAmount);

	UFUNCTION(BlueprintCallable, Category = "Combat|Health")
	void Kill();

	UFUNCTION(BlueprintCallable, Category = "Combat|Health")
	void Respawn();

	//--------------------------------------------------------------------------
	// STAMINA SYSTEM
	//--------------------------------------------------------------------------

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Stamina")
	float MaxStamina = 100.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Stamina")
	float CurrentStamina = 100.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Stamina")
	float StaminaRegenRate = 10.0f;

	UFUNCTION(BlueprintCallable, Category = "Combat|Stamina")
	bool ConsumeStamina(float Amount);

	//--------------------------------------------------------------------------
	// SHIELD SYSTEM
	//--------------------------------------------------------------------------

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Shield")
	float MaxShield = 50.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Shield")
	float CurrentShield = 0.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Shield")
	float ShieldRegenRate = 5.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Shield")
	float ShieldRegenDelay = 3.0f;

	//--------------------------------------------------------------------------
	// WEAPON SYSTEM
	//--------------------------------------------------------------------------

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Weapon")
	int32 WeaponType = 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Weapon")
	float MaxAmmo = 100.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Weapon")
	float CurrentAmmo = 100.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Weapon")
	float WeaponCooldownDuration = 0.5f;

	UPROPERTY(BlueprintReadOnly, Category = "Combat|Weapon")
	float CurrentWeaponCooldown = 0.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Weapon")
	float AmmoCostPerShot = 1.0f;

	UFUNCTION(BlueprintCallable, Category = "Combat|Weapon")
	bool FireWeapon();

	UFUNCTION(BlueprintCallable, Category = "Combat|Weapon")
	void ReloadWeapon();

private:
	/** Time since last damage (for shield regen) */
	float TimeSinceLastDamage = 0.0f;

	/** Is character dead? */
	bool bIsDead = false;

	/** Update timers (cooldowns, regeneration) */
	void UpdateTimers(float DeltaTime);
	void UpdateShieldRegeneration(float DeltaTime);
	void UpdateStaminaRegeneration(float DeltaTime);
	void UpdateWeaponCooldown(float DeltaTime);
};
