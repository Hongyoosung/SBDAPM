#include "Core/SimulationManagerGameMode.h"
#include "Team/TeamLeaderComponent.h"
#include "DrawDebugHelpers.h"
#include "Engine/World.h"

ASimulationManagerGameMode::ASimulationManagerGameMode()
{
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.TickInterval = 0.1f; // 10 Hz
}

void ASimulationManagerGameMode::BeginPlay()
{
	Super::BeginPlay();

	if (bAutoStartSimulation)
	{
		StartSimulation();
	}
}

void ASimulationManagerGameMode::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (bSimulationRunning)
	{
		UpdateStatistics();

		if (bDrawDebugInfo)
		{
			float CurrentTime = GetWorld()->GetTimeSeconds();
			if (CurrentTime - LastDebugDrawTime >= DebugDrawInterval)
			{
				DrawDebugInformation();
				LastDebugDrawTime = CurrentTime;
			}
		}
	}
}

//------------------------------------------------------------------------------
// TEAM REGISTRATION
//------------------------------------------------------------------------------

bool ASimulationManagerGameMode::RegisterTeam(
	int32 TeamID,
	UTeamLeaderComponent* TeamLeader,
	const FString& TeamName,
	FLinearColor TeamColor)
{
	if (RegisteredTeams.Contains(TeamID))
	{
		UE_LOG(LogTemp, Warning, TEXT("SimulationManager: Team %d already registered"), TeamID);
		return false;
	}

	if (!TeamLeader)
	{
		UE_LOG(LogTemp, Error, TEXT("SimulationManager: Cannot register team %d - TeamLeader is null"), TeamID);
		return false;
	}

	FTeamInfo NewTeam;
	NewTeam.TeamID = TeamID;
	NewTeam.TeamName = TeamName;
	NewTeam.TeamLeader = TeamLeader;
	NewTeam.TeamColor = TeamColor;
	NewTeam.bIsActive = bSimulationRunning;

	RegisteredTeams.Add(TeamID, NewTeam);

	UE_LOG(LogTemp, Log, TEXT("SimulationManager: Registered team %d (%s)"), TeamID, *TeamName);
	return true;
}

void ASimulationManagerGameMode::UnregisterTeam(int32 TeamID)
{
	if (!RegisteredTeams.Contains(TeamID))
	{
		return;
	}

	// Remove all team members from actor map
	FTeamInfo& TeamInfo = RegisteredTeams[TeamID];
	for (AActor* Member : TeamInfo.TeamMembers)
	{
		ActorToTeamMap.Remove(Member);
	}

	// Remove team
	RegisteredTeams.Remove(TeamID);

	UE_LOG(LogTemp, Log, TEXT("SimulationManager: Unregistered team %d"), TeamID);
}

bool ASimulationManagerGameMode::RegisterTeamMember(int32 TeamID, AActor* Agent)
{
	if (!Agent)
	{
		return false;
	}

	FTeamInfo* TeamInfo = RegisteredTeams.Find(TeamID);
	if (!TeamInfo)
	{
		UE_LOG(LogTemp, Warning, TEXT("SimulationManager: Cannot register member - team %d not found"), TeamID);
		return false;
	}

	// Remove from old team if exists
	int32* OldTeamID = ActorToTeamMap.Find(Agent);
	if (OldTeamID)
	{
		UnregisterTeamMember(*OldTeamID, Agent);
	}

	// Add to new team
	TeamInfo->TeamMembers.AddUnique(Agent);
	ActorToTeamMap.Add(Agent, TeamID);

	UE_LOG(LogTemp, Log, TEXT("SimulationManager: Registered %s to team %d"), *Agent->GetName(), TeamID);
	return true;
}

void ASimulationManagerGameMode::UnregisterTeamMember(int32 TeamID, AActor* Agent)
{
	if (!Agent)
	{
		return;
	}

	FTeamInfo* TeamInfo = RegisteredTeams.Find(TeamID);
	if (TeamInfo)
	{
		TeamInfo->TeamMembers.Remove(Agent);
	}

	ActorToTeamMap.Remove(Agent);
}

int32 ASimulationManagerGameMode::GetTeamIDForActor(AActor* Agent) const
{
	if (!Agent)
	{
		return -1;
	}

	const int32* TeamID = ActorToTeamMap.Find(Agent);
	return TeamID ? *TeamID : -1;
}

//------------------------------------------------------------------------------
// ENEMY TEAM MANAGEMENT
//------------------------------------------------------------------------------

void ASimulationManagerGameMode::SetEnemyTeams(int32 TeamID, const TArray<int32>& EnemyTeamIDs)
{
	FTeamInfo* TeamInfo = RegisteredTeams.Find(TeamID);
	if (!TeamInfo)
	{
		return;
	}

	TeamInfo->EnemyTeamIDs.Empty();
	for (int32 EnemyID : EnemyTeamIDs)
	{
		if (RegisteredTeams.Contains(EnemyID))
		{
			TeamInfo->EnemyTeamIDs.Add(EnemyID);
		}
	}

	UE_LOG(LogTemp, Log, TEXT("SimulationManager: Team %d now has %d enemy teams"),
		TeamID, TeamInfo->EnemyTeamIDs.Num());
}

void ASimulationManagerGameMode::AddEnemyTeam(int32 TeamID, int32 EnemyTeamID)
{
	FTeamInfo* TeamInfo = RegisteredTeams.Find(TeamID);
	if (!TeamInfo || !RegisteredTeams.Contains(EnemyTeamID))
	{
		return;
	}

	TeamInfo->EnemyTeamIDs.Add(EnemyTeamID);
	UE_LOG(LogTemp, Log, TEXT("SimulationManager: Team %d added enemy team %d"), TeamID, EnemyTeamID);
}

void ASimulationManagerGameMode::RemoveEnemyTeam(int32 TeamID, int32 EnemyTeamID)
{
	FTeamInfo* TeamInfo = RegisteredTeams.Find(TeamID);
	if (!TeamInfo)
	{
		return;
	}

	TeamInfo->EnemyTeamIDs.Remove(EnemyTeamID);
}

void ASimulationManagerGameMode::SetMutualEnemies(int32 TeamID1, int32 TeamID2)
{
	if (!RegisteredTeams.Contains(TeamID1) || !RegisteredTeams.Contains(TeamID2))
	{
		return;
	}

	AddEnemyTeam(TeamID1, TeamID2);
	AddEnemyTeam(TeamID2, TeamID1);

	UE_LOG(LogTemp, Log, TEXT("SimulationManager: Teams %d and %d are now mutual enemies"), TeamID1, TeamID2);
}

bool ASimulationManagerGameMode::AreTeamsEnemies(int32 TeamID1, int32 TeamID2) const
{
	const FTeamInfo* Team1 = RegisteredTeams.Find(TeamID1);
	if (!Team1)
	{
		return false;
	}

	return Team1->EnemyTeamIDs.Contains(TeamID2);
}

bool ASimulationManagerGameMode::AreActorsEnemies(AActor* Actor1, AActor* Actor2) const
{
	int32 Team1 = GetTeamIDForActor(Actor1);
	int32 Team2 = GetTeamIDForActor(Actor2);

	if (Team1 == -1 || Team2 == -1)
	{
		return false;
	}

	return AreTeamsEnemies(Team1, Team2);
}

TArray<int32> ASimulationManagerGameMode::GetEnemyTeamIDs(int32 TeamID) const
{
	const FTeamInfo* TeamInfo = RegisteredTeams.Find(TeamID);
	if (!TeamInfo)
	{
		return TArray<int32>();
	}

	return TeamInfo->EnemyTeamIDs.Array();
}

TArray<AActor*> ASimulationManagerGameMode::GetEnemyActors(int32 TeamID) const
{
	TArray<AActor*> EnemyActors;

	const FTeamInfo* TeamInfo = RegisteredTeams.Find(TeamID);
	if (!TeamInfo)
	{
		return EnemyActors;
	}

	for (int32 EnemyTeamID : TeamInfo->EnemyTeamIDs)
	{
		const FTeamInfo* EnemyTeam = RegisteredTeams.Find(EnemyTeamID);
		if (EnemyTeam)
		{
			EnemyActors.Append(EnemyTeam->TeamMembers);
		}
	}

	return EnemyActors;
}

//------------------------------------------------------------------------------
// TEAM QUERIES
//------------------------------------------------------------------------------

bool ASimulationManagerGameMode::GetTeamInfo(int32 TeamID, FTeamInfo& OutTeamInfo) const
{
	const FTeamInfo* TeamInfo = RegisteredTeams.Find(TeamID);
	if (!TeamInfo)
	{
		return false;
	}

	OutTeamInfo = *TeamInfo;
	return true;
}

UTeamLeaderComponent* ASimulationManagerGameMode::GetTeamLeader(int32 TeamID) const
{
	const FTeamInfo* TeamInfo = RegisteredTeams.Find(TeamID);
	return TeamInfo ? TeamInfo->TeamLeader : nullptr;
}

TArray<AActor*> ASimulationManagerGameMode::GetTeamMembers(int32 TeamID) const
{
	const FTeamInfo* TeamInfo = RegisteredTeams.Find(TeamID);
	return TeamInfo ? TeamInfo->TeamMembers : TArray<AActor*>();
}

TArray<int32> ASimulationManagerGameMode::GetAllTeamIDs() const
{
	TArray<int32> TeamIDs;
	RegisteredTeams.GetKeys(TeamIDs);
	return TeamIDs;
}

bool ASimulationManagerGameMode::IsTeamRegistered(int32 TeamID) const
{
	return RegisteredTeams.Contains(TeamID);
}

//------------------------------------------------------------------------------
// SIMULATION CONTROL
//------------------------------------------------------------------------------

void ASimulationManagerGameMode::StartSimulation()
{
	bSimulationRunning = true;
	SimulationStartTime = GetWorld()->GetTimeSeconds();

	for (auto& Pair : RegisteredTeams)
	{
		Pair.Value.bIsActive = true;
	}

	UE_LOG(LogTemp, Log, TEXT("SimulationManager: Simulation started with %d teams"), RegisteredTeams.Num());
}

void ASimulationManagerGameMode::StopSimulation()
{
	bSimulationRunning = false;

	for (auto& Pair : RegisteredTeams)
	{
		Pair.Value.bIsActive = false;
	}

	UE_LOG(LogTemp, Log, TEXT("SimulationManager: Simulation stopped"));
}

void ASimulationManagerGameMode::ResetSimulation()
{
	StopSimulation();

	RegisteredTeams.Empty();
	ActorToTeamMap.Empty();
	SimulationStartTime = 0.0f;

	UE_LOG(LogTemp, Log, TEXT("SimulationManager: Simulation reset"));
}

FSimulationStats ASimulationManagerGameMode::GetSimulationStats() const
{
	FSimulationStats Stats;

	Stats.TotalTeams = RegisteredTeams.Num();
	Stats.ActiveTeams = 0;
	Stats.TotalAgents = 0;
	Stats.AliveAgents = 0;

	for (const auto& Pair : RegisteredTeams)
	{
		if (Pair.Value.bIsActive)
		{
			Stats.ActiveTeams++;
		}

		Stats.TotalAgents += Pair.Value.TeamMembers.Num();

		// Count alive agents (non-null and not pending kill)
		for (AActor* Member : Pair.Value.TeamMembers)
		{
			if (Member && IsValid(Member))
			{
				Stats.AliveAgents++;
			}
		}
	}

	if (bSimulationRunning)
	{
		Stats.SimulationTime = GetWorld()->GetTimeSeconds() - SimulationStartTime;
	}

	return Stats;
}

//------------------------------------------------------------------------------
// INTERNAL
//------------------------------------------------------------------------------

void ASimulationManagerGameMode::UpdateStatistics()
{
	// Statistics are computed on-demand in GetSimulationStats()
	// This function can be used for periodic updates if needed
}

void ASimulationManagerGameMode::DrawDebugInformation()
{
	UWorld* World = GetWorld();
	if (!World)
	{
		return;
	}

	FVector DebugOrigin(0, 0, 1000);
	float YOffset = 0.0f;

	// Draw team information
	for (const auto& Pair : RegisteredTeams)
	{
		const FTeamInfo& Team = Pair.Value;
		FString TeamStr = FString::Printf(TEXT("Team %d (%s): %d members, %d enemies"),
			Team.TeamID, *Team.TeamName, Team.TeamMembers.Num(), Team.EnemyTeamIDs.Num());

		DrawDebugString(World, DebugOrigin + FVector(0, 0, YOffset), TeamStr, nullptr,
			Team.TeamColor.ToFColor(true), DebugDrawInterval, false, 1.2f);

		YOffset -= 50.0f;
	}

	// Draw statistics
	FSimulationStats Stats = GetSimulationStats();
	FString StatsStr = FString::Printf(TEXT("Stats: %d/%d teams active, %d/%d agents alive, Time: %.1fs"),
		Stats.ActiveTeams, Stats.TotalTeams, Stats.AliveAgents, Stats.TotalAgents, Stats.SimulationTime);

	DrawDebugString(World, DebugOrigin + FVector(0, 0, YOffset), StatsStr, nullptr,
		FColor::White, DebugDrawInterval, false, 1.5f);
}
