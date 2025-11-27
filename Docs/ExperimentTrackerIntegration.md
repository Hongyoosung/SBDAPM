# ExperimentTracker Integration Guide
**Version:** v3.0 | **Date:** 2025-11-27

---

## Overview

The ExperimentTracker system provides automatic metrics collection and CSV export for academic publication. It integrates seamlessly with your existing SBDAPM system.

---

## Quick Start

### 1. Add to GameMode Blueprint

**Option A: Blueprint (Recommended for Quick Setup)**

1. Open your GameMode Blueprint (`BP_ExperimentGameMode`)
2. Add Component → Search "ExperimentTracker"
3. Set properties in Details panel:
   - Experiment Name: "SelfPlay_Training_4v4"
   - Output Directory: "Experiments"
   - Auto Export CSV: true
   - Track MCTS Metrics: true
   - Track Reward Breakdown: true

**Option B: C++ (For Custom GameMode)**

```cpp
// YourGameMode.h
#include "Core/ExperimentTracker.h"

UCLASS()
class AYourGameMode : public ASimulationManagerGameMode
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    UExperimentTracker* ExperimentTracker;
};

// YourGameMode.cpp
AYourGameMode::AYourGameMode()
{
    ExperimentTracker = CreateDefaultSubobject<UExperimentTracker>(TEXT("ExperimentTracker"));
}
```

---

### 2. Start/End Episodes

**In GameMode BeginPlay:**

```cpp
void AYourGameMode::BeginPlay()
{
    Super::BeginPlay();

    // Configure episode metadata
    FEpisodeMetadata Metadata;
    Metadata.ExperimentName = TEXT("SelfPlay_Training");
    Metadata.MapName = GetWorld()->GetMapName();
    Metadata.TeamAlphaAIType = TEXT("SBDAPM_v3.0");
    Metadata.TeamBravoAIType = TEXT("SBDAPM_v3.0");
    Metadata.EpisodeNumber = CurrentEpisodeNumber;

    // Start tracking
    ExperimentTracker->StartEpisode(Metadata);
}
```

**In GameMode Tick (Check Win Conditions):**

```cpp
void AYourGameMode::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    // Check win conditions every second
    static float TimeSinceLastCheck = 0.0f;
    TimeSinceLastCheck += DeltaTime;

    if (TimeSinceLastCheck >= 1.0f)
    {
        TimeSinceLastCheck = 0.0f;

        // Get team leaders
        UTeamLeaderComponent* AlphaLeader = FindTeamLeader(TEXT("TeamAlpha"));
        UTeamLeaderComponent* BravoLeader = FindTeamLeader(TEXT("TeamBravo"));

        if (!AlphaLeader || !BravoLeader) return;

        // Check elimination win condition
        int32 AlphaAlive = AlphaLeader->GetAliveFollowers().Num();
        int32 BravoAlive = BravoLeader->GetAliveFollowers().Num();

        FString Winner = TEXT("Draw");
        FString WinCondition = TEXT("None");

        if (AlphaAlive == 0 && BravoAlive > 0)
        {
            Winner = TEXT("TeamBravo");
            WinCondition = TEXT("Elimination");
            EndEpisode(Winner, WinCondition);
        }
        else if (BravoAlive == 0 && AlphaAlive > 0)
        {
            Winner = TEXT("TeamAlpha");
            WinCondition = TEXT("Elimination");
            EndEpisode(Winner, WinCondition);
        }
        else if (AlphaAlive == 0 && BravoAlive == 0)
        {
            Winner = TEXT("Draw");
            WinCondition = TEXT("MutualElimination");
            EndEpisode(Winner, WinCondition);
        }

        // Check timeout (10 minutes)
        float EpisodeTime = GetWorld()->GetTimeSeconds() - EpisodeStartTime;
        if (EpisodeTime >= 600.0f)
        {
            Winner = (AlphaAlive > BravoAlive) ? TEXT("TeamAlpha") :
                     (BravoAlive > AlphaAlive) ? TEXT("TeamBravo") : TEXT("Draw");
            WinCondition = TEXT("Timeout");
            EndEpisode(Winner, WinCondition);
        }
    }
}

void AYourGameMode::EndEpisode(const FString& Winner, const FString& WinCondition)
{
    // End tracking and export
    ExperimentTracker->EndEpisode(Winner, WinCondition);

    // Export experiences
    ExportAllExperiences();

    // Optionally restart level for next episode
    // UGameplayStatics::OpenLevel(this, FName(*GetWorld()->GetName()), true);
}
```

---

### 3. Record Events (Automatic via Delegates)

**Option A: Bind to Existing Events**

```cpp
// In BeginPlay, bind to HealthComponent events
void AYourGameMode::BindHealthEvents(AActor* Agent)
{
    UHealthComponent* HealthComp = Agent->FindComponentByClass<UHealthComponent>();
    if (!HealthComp) return;

    // Bind kill event
    HealthComp->OnKillConfirmed.AddDynamic(this, &AYourGameMode::OnAgentKill);

    // Bind death event
    HealthComp->OnDeath.AddDynamic(this, &AYourGameMode::OnAgentDeath);

    // Bind damage events
    HealthComp->OnDamageDealt.AddDynamic(this, &AYourGameMode::OnDamageDealt);
    HealthComp->OnDamageTaken.AddDynamic(this, &AYourGameMode::OnDamageTaken);
}

void AYourGameMode::OnAgentKill(AActor* Killer, AActor* Victim)
{
    FString KillerTeam = GetActorTeam(Killer);
    FString VictimTeam = GetActorTeam(Victim);

    // Check if coordinated kill (during strategic command)
    bool bCoordinated = IsCoordinatedAction(Killer);

    ExperimentTracker->RecordKill(KillerTeam, Killer, Victim, bCoordinated);
}

void AYourGameMode::OnAgentDeath(AActor* Victim)
{
    FString Team = GetActorTeam(Victim);
    ExperimentTracker->RecordDeath(Team, Victim);
}

void AYourGameMode::OnDamageDealt(AActor* Dealer, float Damage)
{
    FString Team = GetActorTeam(Dealer);
    ExperimentTracker->RecordDamage(Team, Damage, true); // bDealt = true
}

void AYourGameMode::OnDamageTaken(AActor* Victim, float Damage)
{
    FString Team = GetActorTeam(Victim);
    ExperimentTracker->RecordDamage(Team, Damage, false); // bDealt = false
}
```

**Option B: Manual Recording (Simpler but Less Automated)**

Add calls to `RecordXXX()` methods wherever events occur:

```cpp
// When objective completed
ExperimentTracker->RecordObjectiveEvent(TEXT("TeamAlpha"), true, CompletionTime);

// When MCTS executes (add to TeamLeaderComponent::OnObjectiveMCTSComplete)
ExperimentTracker->RecordMCTSExecution(
    TeamName,
    ExecutionTime,
    ValueVariance,
    PolicyEntropy
);

// When combined fire detected (add to RewardCalculator::RegisterCombinedFire)
ExperimentTracker->RecordCombinedFire(TeamName);
```

---

### 4. Modify TeamLeaderComponent (Optional Enhancement)

To auto-report MCTS metrics, add this to `TeamLeaderComponent.cpp`:

```cpp
// At end of OnObjectiveMCTSComplete (line 896)
void UTeamLeaderComponent::OnObjectiveMCTSComplete(TMap<AActor*, UObjective*> NewObjectives)
{
    // ... existing code ...

    // Report to ExperimentTracker
    if (UWorld* World = GetWorld())
    {
        if (ASimulationManagerGameMode* GM = Cast<ASimulationManagerGameMode>(World->GetAuthGameMode()))
        {
            if (UExperimentTracker* Tracker = GM->FindComponentByClass<UExperimentTracker>())
            {
                Tracker->RecordMCTSExecution(
                    TeamName,
                    ExecutionTime,
                    ValueVariance,
                    PolicyEntropy
                );

                Tracker->RecordFormationCoherence(
                    TeamName,
                    CurrentTeamObservation.FormationCoherence
                );
            }
        }
    }
}
```

---

## CSV Output Format

**File Location:** `Saved/Experiments/[ExperimentName]_[MapName].csv`

**Columns:**

```
EpisodeNumber, Timestamp, MapName, Winner, WinCondition, Duration,
Alpha_Kills, Alpha_Deaths, Alpha_KD, Alpha_DamageDealt, Alpha_DamageTaken,
Alpha_CoordKills, Alpha_CombinedFire, Alpha_FormationCoherence,
Alpha_ObjectivesComplete, Alpha_ObjectivesFailed,
Alpha_MCTSExecutions, Alpha_MCTSLatency, Alpha_ValueVariance, Alpha_PolicyEntropy,
Alpha_TotalReward, Alpha_IndividualReward, Alpha_CoordinationReward, Alpha_StrategicReward,
[... same for Bravo ...]
AvgFPS, PeakMemoryMB
```

**Example Row:**

```csv
1,2025-11-27_14-30-15,Training_BasicCombat_4v4,TeamAlpha,Elimination,45.2,
12,8,1.50,300,200,4,3,0.65,2,0,15,42.5,0.12,1.8,125.5,80.0,30.5,15.0,
8,12,0.67,200,300,2,1,0.58,1,1,14,38.2,0.15,1.6,85.3,60.0,15.3,10.0,
58.3,1024.5
```

---

## Analysis Workflow

### Step 1: Run Experiments

1. Load training map in UE5
2. PIE (Play In Editor)
3. Episodes run automatically
4. CSV appended after each episode

### Step 2: Analyze Results

```bash
# Install Python dependencies
pip install -r Scripts/requirements_analysis.txt

# Analyze single experiment
python Scripts/analyze_experiments.py --csv Saved/Experiments/SelfPlay_Training_BasicCombat_4v4.csv

# Analyze multiple experiments (wildcard)
python Scripts/analyze_experiments.py --csv "Saved/Experiments/SelfPlay_*.csv" --output Results/SelfPlay/

# Analyze baseline comparison
python Scripts/analyze_experiments.py --csv "Saved/Experiments/Baseline_*.csv" --output Results/Baseline/
```

### Step 3: View Results

Generated files in `Results/`:
- `win_rates.png` - Win rate over time
- `kd_comparison.png` - K/D ratio comparison
- `coordination_metrics.png` - Coordination metrics
- `mcts_performance.png` - MCTS latency/uncertainty
- `reward_breakdown.png` - Hierarchical reward breakdown
- `episode_duration.png` - Episode length analysis
- `summary_report.txt` - Text summary

---

## Example: Complete Integration

**BP_ExperimentGameMode.cpp (Minimal Setup)**

```cpp
#include "Core/ExperimentTracker.h"
#include "Team/TeamLeaderComponent.h"
#include "Team/FollowerAgentComponent.h"

void ABP_ExperimentGameMode::BeginPlay()
{
    Super::BeginPlay();

    // Find tracker component
    ExperimentTracker = FindComponentByClass<UExperimentTracker>();
    if (!ExperimentTracker)
    {
        UE_LOG(LogTemp, Error, TEXT("ExperimentTracker not found! Add to GameMode Blueprint"));
        return;
    }

    // Start episode
    FEpisodeMetadata Meta;
    Meta.ExperimentName = TEXT("SelfPlay_Training");
    Meta.MapName = GetWorld()->GetMapName();
    Meta.TeamAlphaAIType = TEXT("SBDAPM_v3.0");
    Meta.TeamBravoAIType = TEXT("SBDAPM_v3.0");

    ExperimentTracker->StartEpisode(Meta);

    // Register team leaders (for auto-metrics)
    TArray<AActor*> Leaders;
    UGameplayStatics::GetAllActorsWithTag(GetWorld(), TEXT("TeamLeader"), Leaders);
    for (AActor* LeaderActor : Leaders)
    {
        if (UTeamLeaderComponent* Leader = LeaderActor->FindComponentByClass<UTeamLeaderComponent>())
        {
            ExperimentTracker->RegisterTeamLeader(Leader);
        }
    }

    EpisodeStartTime = GetWorld()->GetTimeSeconds();
}

void ABP_ExperimentGameMode::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    CheckWinConditions();
}

void ABP_ExperimentGameMode::CheckWinConditions()
{
    // ... (see above for full implementation)
}
```

---

## Troubleshooting

### CSV Not Generated
- Check `bAutoExportCSV = true` in ExperimentTracker properties
- Verify `EndEpisode()` is called
- Check logs for "📊 [EXPERIMENT] Episode ENDED"
- Ensure `Saved/Experiments/` directory exists (auto-created)

### Missing Metrics
- `bTrackMCTSMetrics = true` → Enables MCTS columns
- `bTrackRewardBreakdown = true` → Enables reward columns
- Call `RecordXXX()` methods for specific events

### Python Script Errors
- Install dependencies: `pip install -r Scripts/requirements_analysis.txt`
- Verify CSV file path is correct
- Check CSV header matches expected format

---

## Academic Publication Checklist

- [ ] Run 100+ self-play episodes per curriculum level
- [ ] Run 100+ evaluation episodes vs each baseline
- [ ] Export CSVs for all experiments
- [ ] Generate plots with `analyze_experiments.py`
- [ ] Calculate statistical significance (t-test, p < 0.05)
- [ ] Create comparison tables (Win Rate, K/D, Coordination %)
- [ ] Include MCTS performance metrics (latency, uncertainty)
- [ ] Document emergent behaviors (video capture + screenshots)
- [ ] Create ablation study results (MCTS only, RL only, etc.)

---

## References

- **ExperimentTracker.h:69-153** - API documentation
- **LevelDesignTemplate.md** - Level setup guide
- **analyze_experiments.py** - Analysis script
- **CLAUDE.md:349-366** - Success metrics targets

---

**Quick Command Reference:**

```bash
# Run analysis
python Scripts/analyze_experiments.py --csv "Saved/Experiments/*.csv" --output Results/

# View summary
cat Results/summary_report.txt

# Compare experiments
python Scripts/analyze_experiments.py \
  --csv "Saved/Experiments/SelfPlay_*.csv" \
  --csv "Saved/Experiments/Baseline_*.csv" \
  --output Results/Comparison/
```

---

**End of Integration Guide**
