// Copyright Epic Games, Inc. All Rights Reserved.

#include "Combat/ProjectileBase.h"
#include "Combat/HealthComponent.h"
#include "Components/SphereComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Particles/ParticleSystemComponent.h"
#include "GameFramework/ProjectileMovementComponent.h"
#include "DrawDebugHelpers.h"
#include "Engine/World.h"
#include "TimerManager.h"
#include "Kismet/GameplayStatics.h"
#include "Core/SimulationManagerGameMode.h"

AProjectileBase::AProjectileBase()
{
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.TickGroup = TG_PostPhysics;

	// Create collision component
	CollisionComponent = CreateDefaultSubobject<USphereComponent>(TEXT("CollisionComponent"));
	CollisionComponent->InitSphereRadius(CollisionRadius);
	CollisionComponent->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
	CollisionComponent->SetCollisionObjectType(ECC_WorldDynamic);
	CollisionComponent->SetCollisionResponseToAllChannels(ECR_Ignore);
	CollisionComponent->SetCollisionResponseToChannel(ECC_Pawn, ECR_Block);
	CollisionComponent->SetCollisionResponseToChannel(ECC_WorldStatic, ECR_Block);
	CollisionComponent->SetCollisionResponseToChannel(ECC_WorldDynamic, ECR_Block);
	CollisionComponent->SetNotifyRigidBodyCollision(true);
	RootComponent = CollisionComponent;

	// Create projectile movement
	ProjectileMovement = CreateDefaultSubobject<UProjectileMovementComponent>(TEXT("ProjectileMovement"));
	ProjectileMovement->UpdatedComponent = CollisionComponent;
	ProjectileMovement->InitialSpeed = 0.0f; // Start at zero, set in InitializeProjectile
	ProjectileMovement->MaxSpeed = ProjectileSpeed;
	ProjectileMovement->bRotationFollowsVelocity = true;
	ProjectileMovement->bShouldBounce = false;
	ProjectileMovement->ProjectileGravityScale = 0.0f; // No gravity by default
	ProjectileMovement->bInitialVelocityInLocalSpace = false;
	ProjectileMovement->bAutoActivate = true; // Ensure component is active

	// Create mesh component (optional)
	MeshComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshComponent"));
	MeshComponent->SetupAttachment(CollisionComponent);
	MeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);

	// Create trail VFX component (optional)
	TrailVFX = CreateDefaultSubobject<UParticleSystemComponent>(TEXT("TrailVFX"));
	TrailVFX->SetupAttachment(CollisionComponent);
	TrailVFX->bAutoActivate = false;

	// Set lifetime
	InitialLifeSpan = 0.0f; // We handle lifetime manually for better control
}

void AProjectileBase::BeginPlay()
{
	Super::BeginPlay();

	// Bind hit event
	if (CollisionComponent)
	{
		CollisionComponent->OnComponentHit.AddDynamic(this, &AProjectileBase::OnProjectileHit);
		CollisionComponent->SetSphereRadius(CollisionRadius);
	}

	// Set projectile max speed
	if (ProjectileMovement)
	{
		ProjectileMovement->MaxSpeed = ProjectileSpeed;
	}

	// Initialize state
	SpawnLocation = GetActorLocation();
	LastFrameLocation = SpawnLocation;
	DistanceTraveled = 0.0f;
	HitCount = 0;
	bIsActive = true;
	HitActors.Empty();

	// Start lifetime timer
	if (Lifetime > 0.0f)
	{
		GetWorldTimerManager().SetTimer(LifetimeTimerHandle, this, &AProjectileBase::OnLifetimeExpired, Lifetime, false);
	}

	// Activate trail VFX
	if (TrailVFX && TrailVFX->Template)
	{
		TrailVFX->Activate(true);
	}
}

void AProjectileBase::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (!bIsActive)
	{
		return;
	}

	// Update distance traveled
	FVector CurrentLocation = GetActorLocation();
	float FrameDistance = FVector::Dist(LastFrameLocation, CurrentLocation);
	DistanceTraveled += FrameDistance;
	LastFrameLocation = CurrentLocation;

	// Update homing
	if (bEnableHoming && HomingTarget && ProjectileMovement)
	{
		ProjectileMovement->bIsHomingProjectile = true;
		ProjectileMovement->HomingTargetComponent = HomingTarget->GetRootComponent();
		ProjectileMovement->HomingAccelerationMagnitude = HomingAcceleration;
	}

	// Debug drawing
	if (bEnableDebugDrawing)
	{
		DrawDebugSphere(GetWorld(), CurrentLocation, CollisionRadius, 8, FColor::Yellow, false, 0.0f, 0, 1.0f);
		DrawDebugLine(GetWorld(), SpawnLocation, CurrentLocation, FColor::Cyan, false, 0.0f, 0, 1.0f);

		FString DebugText = FString::Printf(TEXT("Dist: %.0fcm\nDmg: %.1f"), DistanceTraveled, CalculateDamageWithFalloff(DistanceTraveled));
		DrawDebugString(GetWorld(), CurrentLocation + FVector(0, 0, 30), DebugText, nullptr, FColor::White, 0.0f, true);
	}
}

//------------------------------------------------------------------------------
// PUBLIC API
//------------------------------------------------------------------------------

void AProjectileBase::InitializeProjectile(AActor* InOwner, AActor* InInstigator, float InBaseDamage, const FVector& InDirection)
{
	OwnerActor = InOwner;
	InstigatorActor = InInstigator;
	BaseDamage = InBaseDamage;

	if (CollisionComponent)
	{
		// 투사체가 이동할 때 Owner(캐릭터)를 무시하도록 설정
		if (InOwner)
		{
			CollisionComponent->IgnoreActorWhenMoving(InOwner, true);

			// (선택 사항) 만약 Owner 측에서도 투사체를 확실히 무시하게 하려면:
			// Owner의 루트 컴포넌트(주로 캡슐)를 가져와서 설정해야 합니다.
			if (UPrimitiveComponent* OwnerRoot = Cast<UPrimitiveComponent>(InOwner->GetRootComponent()))
			{
				OwnerRoot->IgnoreActorWhenMoving(this, true);
			}
		}

		// Instigator(가해자)도 무시
		/*if (InInstigator && InInstigator != InOwner)
		{
			CollisionComponent->IgnoreActorWhenMoving(InInstigator, true);

			if (UPrimitiveComponent* InstigatorRoot = Cast<UPrimitiveComponent>(InInstigator->GetRootComponent()))
			{
				InstigatorRoot->IgnoreActorWhenMoving(this, true);
			}
		}*/
	}

	// 2. 속도 및 이동 설정
	if (ProjectileMovement)
	{
		FVector NormalizedDir = InDirection.GetSafeNormal();

		// 속도 값 강제 설정
		ProjectileMovement->InitialSpeed = ProjectileSpeed;
		ProjectileMovement->MaxSpeed = ProjectileSpeed;

		// 속도 벡터 직접 할당 (중요)
		ProjectileMovement->Velocity = NormalizedDir * ProjectileSpeed;

		// 컴포넌트 활성화 및 업데이트 대상 설정
		ProjectileMovement->SetUpdatedComponent(CollisionComponent);
		ProjectileMovement->UpdateComponentVelocity();
	}

	SetActorRotation(InDirection.Rotation());
}

void AProjectileBase::SetVelocity(const FVector& NewVelocity)
{
	if (ProjectileMovement)
	{
		ProjectileMovement->Velocity = NewVelocity;
		SetActorRotation(NewVelocity.Rotation());
	}
}

float AProjectileBase::GetRemainingLifetime() const
{
	if (Lifetime <= 0.0f)
	{
		return 999999.0f;
	}

	float Elapsed = GetWorldTimerManager().GetTimerElapsed(LifetimeTimerHandle);
	return FMath::Max(Lifetime - Elapsed, 0.0f);
}

void AProjectileBase::Explode()
{
	if (!bIsActive)
	{
		return;
	}

	//UE_LOG(LogTemp, Warning, TEXT("💥 Projectile exploded at %s"), *GetActorLocation().ToString());

	SpawnImpactEffects(GetActorLocation(), FVector::UpVector);
	DeactivateProjectile();
}

//------------------------------------------------------------------------------
// COLLISION & DAMAGE
//------------------------------------------------------------------------------

void AProjectileBase::OnProjectileHit(UPrimitiveComponent* HitComponent, AActor* OtherActor,
	UPrimitiveComponent* OtherComp, FVector NormalImpulse, const FHitResult& Hit)
{
	if (!bIsActive || !OtherActor)
	{
		return;
	}

	// Ignore self and instigator
	if (OtherActor == this || OtherActor == OwnerActor || OtherActor == InstigatorActor)
	{
		return;
	}

	// Check if already hit this actor (for penetration)
	if (HitActors.Contains(OtherActor))
	{
		return;
	}

	// Validate hit
	if (!ValidateHit(OtherActor))
	{
		UE_LOG(LogTemp, Verbose, TEXT("❌ Invalid hit: %s (friendly fire disabled)"), *OtherActor->GetName());
		return;
	}


	// Get hit location and normal from HitResult
	FVector HitLocation = Hit.ImpactPoint;
	FVector HitNormal = Hit.ImpactNormal;

	// Apply damage
	ApplyDamageToActor(OtherActor, HitLocation, HitNormal);

	// Track hit
	HitActors.Add(OtherActor);
	HitCount++;
	
	// Spawn impact effects
	SpawnImpactEffects(HitLocation, HitNormal);

	// Create hit data
	float FinalDamage = CalculateDamageWithFalloff(DistanceTraveled);
	FProjectileHitData HitData(OtherActor, HitLocation, HitNormal, DistanceTraveled, FinalDamage, true);

	// Broadcast event
	OnProjectileHit_Delegate.Broadcast(HitData);

	// Check penetration
	if (bPenetrateActors && (MaxPenetrations == 0 || HitCount < MaxPenetrations))
	{
		UE_LOG(LogTemp, Log, TEXT("🔫 Projectile penetrated %s (Hit %d/%d)"),
			*OtherActor->GetName(), HitCount, MaxPenetrations);
		return; // Continue flying
	}

	// Destroy or deactivate
	if (bDestroyOnHit)
	{
		DeactivateProjectile();
	}
}

void AProjectileBase::OnLifetimeExpired()
{
	if (!bIsActive)
	{
		return;
	}

	/*UE_LOG(LogTemp, Verbose, TEXT("Projectile lifetime expired (%.1fs, %.0fcm traveled)"),
		Lifetime, DistanceTraveled);*/

	OnProjectileExpired_Delegate.Broadcast();
	DeactivateProjectile();
}

//------------------------------------------------------------------------------
// DAMAGE CALCULATION
//------------------------------------------------------------------------------

float AProjectileBase::CalculateDamageWithFalloff(float Distance) const
{
	if (!bEnableDamageFalloff || Distance <= FalloffStartDistance)
	{
		return BaseDamage;
	}

	if (Distance >= FalloffEndDistance)
	{
		return BaseDamage * MinDamageMultiplier;
	}

	// Linear interpolation between start and end
	float FalloffRange = FalloffEndDistance - FalloffStartDistance;
	float DistanceIntoFalloff = Distance - FalloffStartDistance;
	float FalloffAlpha = DistanceIntoFalloff / FalloffRange;

	float DamageMultiplier = FMath::Lerp(1.0f, MinDamageMultiplier, FalloffAlpha);
	return BaseDamage * DamageMultiplier;
}

bool AProjectileBase::ValidateHit(AActor* HitActor) const
{
	if (!HitActor)
	{
		return false;
	}

	// Check friendly fire using SimulationManager team system
	if (!bAllowFriendlyFire && OwnerActor)
	{
		// Get SimulationManager for proper team checking
		if (UWorld* World = GetWorld())
		{
			if (ASimulationManagerGameMode* SimManager = Cast<ASimulationManagerGameMode>(World->GetAuthGameMode()))
			{
				// If actors are NOT enemies (same team or neutral), block hit
				if (!SimManager->AreActorsEnemies(OwnerActor, HitActor))
				{
					return false;
				}
			}
		}
	}

	return true;
}

void AProjectileBase::ApplyDamageToActor(AActor* HitActor, const FVector& HitLocation, const FVector& HitNormal)
{
	if (!HitActor)
	{
		UE_LOG(LogTemp, Warning, TEXT("⚠️ ApplyDamageToActor called with null HitActor"));
		
		return;
	}

	// Calculate final damage
	float FinalDamage = CalculateDamageWithFalloff(DistanceTraveled);

	

	// Find HealthComponent on hit actor
	UHealthComponent* TargetHealth = HitActor->FindComponentByClass<UHealthComponent>();
	if (TargetHealth)
	{
		// Apply damage via HealthComponent
		
		float ActualDamage = TargetHealth->TakeDamage(FinalDamage, InstigatorActor, OwnerActor, HitLocation, HitNormal);
		
		// Notify our owner's HealthComponent that we dealt damage
		if (OwnerActor)
		{
			UHealthComponent* OwnerHealth = OwnerActor->FindComponentByClass<UHealthComponent>();
			if (OwnerHealth)
			{
				OwnerHealth->NotifyDamageDealt(HitActor, ActualDamage);
			}
			else
			{
				UE_LOG(LogTemp, Warning, TEXT("⚠️ Owner %s has no HealthComponent to notify damage dealt"), *OwnerActor->GetName());
			}
		}
		else
		{
			UE_LOG(LogTemp, Warning, TEXT("⚠️ Projectile has no OwnerActor to notify damage dealt"));
		}

		/*UE_LOG(LogTemp, Log, TEXT("💥 Projectile hit %s → Damage: %.1f (Distance: %.0fcm)"),
			*HitActor->GetName(), ActualDamage, DistanceTraveled);*/
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("⚠️ Projectile hit %s (no HealthComponent)"), *HitActor->GetName());
	}
}

//------------------------------------------------------------------------------
// VFX/SFX
//------------------------------------------------------------------------------

void AProjectileBase::SpawnImpactEffects(const FVector& ImpactLocation, const FVector& ImpactNormal)
{
	// Spawn impact VFX
	if (ImpactVFX)
	{
		UGameplayStatics::SpawnEmitterAtLocation(GetWorld(), ImpactVFX, ImpactLocation, ImpactNormal.Rotation());
	}

	// Play impact sound
	if (ImpactSound)
	{
		UGameplayStatics::PlaySoundAtLocation(GetWorld(), ImpactSound, ImpactLocation);
	}
}

//------------------------------------------------------------------------------
// LIFECYCLE
//------------------------------------------------------------------------------

void AProjectileBase::DeactivateProjectile()
{
	if (!bIsActive)
	{
		return;
	}

	bIsActive = false;

	// Stop movement
	if (ProjectileMovement)
	{
		ProjectileMovement->Velocity = FVector::ZeroVector;
		ProjectileMovement->Deactivate();
	}

	// Deactivate trail VFX
	if (TrailVFX)
	{
		TrailVFX->Deactivate();
	}

	// Clear lifetime timer
	GetWorldTimerManager().ClearTimer(LifetimeTimerHandle);

	// Destroy actor (or return to pool if pooling is implemented)
	Destroy();
}
