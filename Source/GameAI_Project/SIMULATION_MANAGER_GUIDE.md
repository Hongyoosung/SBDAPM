# Simulation Manager GameMode - Blueprint Setup Guide

Quick guide for setting up multi-team simulations with enemy relationships.

## Setup Steps

### 1. Set GameMode in World Settings
1. Open your level (e.g., `RL_Test_Map`)
2. **Window → World Settings**
3. Set **GameMode Override** to `SimulationManagerGameMode`

### 2. Create Team Setup Actor (Blueprint)

**Option A: Use existing `BP_TeamSetup`**
- Location: `Content/Game/Blueprints/Actor/BP_TeamSetup`

**Option B: Create new Blueprint Actor:**
1. **Content Browser → Add → Blueprint Class → Actor**
2. Name: `BP_TeamSetup`
3. Add component: `TeamLeaderComponent` (if needed per team)

### 3. Register Teams (BeginPlay Event)

```
Event BeginPlay
  ↓
Get Game Mode → Cast to SimulationManagerGameMode
  ↓
Register Team (Team ID: 1, Team Leader: TeamLeader1, Name: "Red Team", Color: Red)
  ↓
Register Team (Team ID: 2, Team Leader: TeamLeader2, Name: "Blue Team", Color: Blue)
  ↓
Set Mutual Enemies (Team ID 1: 1, Team ID 2: 2)
```

**Blueprint Nodes:**
- **Get Game Mode** → `Cast to SimulationManagerGameMode`
- **Register Team** (for each team)
  - `Team ID`: 1, 2, etc. (must be unique)
  - `Team Leader`: Reference to `UTeamLeaderComponent`
  - `Team Name`: "Red Team", "Blue Team"
  - `Team Color`: (R=1,G=0,B=0), (R=0,G=0,B=1)
- **Set Mutual Enemies** (Team ID1=1, Team ID2=2)

### 4. Register Team Members

**For each agent actor (e.g., `BP_FollowerAgent`):**

```
Event BeginPlay
  ↓
Get Game Mode → Cast to SimulationManagerGameMode
  ↓
Register Team Member (Team ID: 1, Agent: Self)
```

**Blueprint Setup:**
1. Add variable `TeamID` (Integer, Editable, Instance Editable)
2. Set per-instance in level (select actor → Details → TeamID = 1 or 2)
3. BeginPlay: Call `Register Team Member` with `TeamID` and `Self`

### 5. Level Setup Example

**Actors in level:**
- `BP_TeamSetup` (1 instance) - Registers teams and enemy relationships
- `BP_RedLeaderAgent` (1 instance) - TeamID = 1
- `BP_FollowerAgent` (N instances) - TeamID = 1 (set individually)
- `BP_BlueLeaderAgent` (1 instance) - TeamID = 2
- `BP_FollowerAgent` (M instances) - TeamID = 2 (set individually)

## Key Functions (Blueprint Callable)

### Team Registration
- `RegisterTeam(TeamID, TeamLeader, TeamName, TeamColor)` - Register a team
- `RegisterTeamMember(TeamID, Agent)` - Add agent to team
- `UnregisterTeam(TeamID)` - Remove team
- `UnregisterTeamMember(TeamID, Agent)` - Remove agent from team

### Enemy Management
- `SetMutualEnemies(TeamID1, TeamID2)` - Both teams become enemies (most common)
- `AddEnemyTeam(TeamID, EnemyTeamID)` - One-way enemy relationship
- `RemoveEnemyTeam(TeamID, EnemyTeamID)` - Remove enemy relationship
- `SetEnemyTeams(TeamID, EnemyTeamIDsArray)` - Set multiple enemies at once

### Queries
- `GetTeamLeader(TeamID)` → Returns `UTeamLeaderComponent*`
- `GetTeamMembers(TeamID)` → Returns `TArray<AActor*>`
- `GetEnemyActors(TeamID)` → Returns all enemy actors across all enemy teams
- `GetTeamIDForActor(Agent)` → Returns team ID (-1 if not found)
- `AreActorsEnemies(Actor1, Actor2)` → Returns bool
- `IsTeamRegistered(TeamID)` → Returns bool

### Simulation Control
- `StartSimulation()` - Enable all teams
- `StopSimulation()` - Disable all teams
- `ResetSimulation()` - Clear all teams and stats
- `IsSimulationRunning()` → Returns bool
- `GetSimulationStats()` → Returns stats (total teams, agents, time)

## Configuration (Editable in GameMode)

**GameMode Settings:**
- `bAutoStartSimulation` (default: true) - Auto-start on BeginPlay
- `bDrawDebugInfo` (default: false) - Enable debug visualization
- `DebugDrawInterval` (default: 1.0s) - Debug update rate

## Common Patterns

### Pattern 1: Red vs Blue (Two-Team)
```
RegisterTeam(1, RedLeader, "Red", Red)
RegisterTeam(2, BlueLeader, "Blue", Blue)
SetMutualEnemies(1, 2)
```

### Pattern 2: Free-For-All (Three Teams)
```
RegisterTeam(1, RedLeader, "Red", Red)
RegisterTeam(2, BlueLeader, "Blue", Blue)
RegisterTeam(3, GreenLeader, "Green", Green)
SetMutualEnemies(1, 2)
SetMutualEnemies(1, 3)
SetMutualEnemies(2, 3)
```

### Pattern 3: Red+Blue vs Green (Alliance)
```
RegisterTeam(1, RedLeader, "Red", Red)
RegisterTeam(2, BlueLeader, "Blue", Blue)
RegisterTeam(3, GreenLeader, "Green", Green)
SetMutualEnemies(1, 3)
SetMutualEnemies(2, 3)
// Teams 1 and 2 are NOT enemies
```

## Accessing from Follower Agents

**In `FollowerAgentComponent` or BT tasks:**

```cpp
// Get enemy actors for my team
ASimulationManagerGameMode* GameMode = GetWorld()->GetAuthGameMode<ASimulationManagerGameMode>();
if (GameMode)
{
    int32 MyTeamID = GameMode->GetTeamIDForActor(GetOwner());
    TArray<AActor*> Enemies = GameMode->GetEnemyActors(MyTeamID);
}
```

**Blueprint equivalent:**
```
Get Game Mode → Cast to SimulationManagerGameMode
  ↓
Get Team ID For Actor (Agent: Self) → MyTeamID
  ↓
Get Enemy Actors (Team ID: MyTeamID) → Enemies Array
```

## Troubleshooting

**Teams not seeing each other as enemies:**
- Verify `SetMutualEnemies` or `AddEnemyTeam` called after both teams registered
- Check team IDs match (use `IsTeamRegistered` to verify)

**Agents not registered:**
- Ensure `RegisterTeamMember` called in agent's BeginPlay AFTER GameMode's BeginPlay
- Use delay or check `IsTeamRegistered` before registering members

**Team leader not found:**
- Verify `TeamLeaderComponent` added to leader actor
- Check component reference passed to `RegisterTeam`

**GetEnemyActors returns empty:**
- Ensure enemy relationships set
- Verify enemy team members registered
- Use `AreActorsEnemies` to debug specific actor pairs

## Performance Notes

- `GetEnemyActors`: O(N×M) where N=enemy teams, M=members per team
- `GetTeamIDForActor`: O(1) - uses cached map
- `AreActorsEnemies`: O(1) - uses cached team IDs and sets
- Recommended: Cache enemy arrays if querying frequently (e.g., every tick)

## Integration with Team Leader

**Team Leader uses GameMode automatically:**
- `TeamLeaderComponent::GetEnemyActors()` → calls `GameMode->GetEnemyActors(MyTeamID)`
- `TeamLeaderComponent::GetTeamMembers()` → calls `GameMode->GetTeamMembers(MyTeamID)`
- See `Team/TeamLeaderComponent.cpp:71-92` for implementation

**No manual integration needed** - Just register teams and members correctly.
