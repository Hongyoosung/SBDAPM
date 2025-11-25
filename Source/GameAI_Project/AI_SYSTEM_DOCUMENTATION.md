# AI System Documentation: AlphaZero-Inspired Multi-Agent Combat AI

**Version:** 3.0
**Engine:** Unreal Engine 5.6
**Language:** C++17
**Last Updated:** 2025-11-26

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Components](#architecture-components)
3. [Team Leader System (Strategic MCTS)](#team-leader-system-strategic-mcts)
4. [Follower Agent System (Tactical RL)](#follower-agent-system-tactical-rl)
5. [Value Network](#value-network)
6. [World Model](#world-model)
7. [Coupled Training](#coupled-training)
8. [Reward System](#reward-system)
9. [Observation System](#observation-system)
10. [Training Pipeline](#training-pipeline)
11. [Performance Characteristics](#performance-characteristics)
12. [Configuration](#configuration)
13. [Troubleshooting](#troubleshooting)

---

## System Overview

### Design Philosophy

The AI system is inspired by AlphaZero but adapted for **real-time, multi-agent, partial observability** combat scenarios. It combines:

- **Strategic Planning**: MCTS tree search for team-level decisions
- **Tactical Execution**: Deep RL for individual agent actions
- **Learned Value Functions**: Neural networks replace hand-crafted heuristics
- **World Model Simulation**: State transition prediction for Monte Carlo rollouts
- **Coupled Training**: MCTS guides RL curriculum, RL provides MCTS priors

### Key Innovations (v3.0)

1. **Value Network**: Estimates team state value → guides MCTS tree search
2. **World Model**: Predicts future states → enables true simulation
3. **Prior-Guided MCTS**: RL policy provides action priors → focuses search
4. **MCTS-Guided Curriculum**: High-uncertainty scenarios prioritized in RL training
5. **Hierarchical Rewards**: Unified reward system aligns strategic + tactical objectives
6. **Continuous Planning**: Proactive MCTS execution (1-2s intervals) with uncertainty quantification
7. **Automated Self-Play**: Complete training pipeline from data collection to evaluation

### System Hierarchy

```
SimulationManagerGameMode (manages teams, enemy relationships)
    ↓
TeamLeaderComponent (per team) - Strategic Layer
    ├─ MCTS Tree Search (continuous, 1.5s intervals)
    │   ├─ ValueNetwork (leaf evaluation)
    │   ├─ WorldModel (state prediction)
    │   └─ RLPolicyNetwork priors (action initialization)
    ├─ Command Generation (strategic commands with confidence)
    ├─ Curriculum Manager (identifies hard scenarios)
    └─ Performance Profiling (tracks MCTS timing)
    ↓
FollowerAgentComponent (per agent) - Tactical Layer
    ├─ Perception (enemy detection, environment raycasting)
    ├─ RLPolicyNetwork (tactical action selection)
    ├─ StateTree (execution framework)
    │   ├─ Tasks (ExecuteAssault, ExecuteDefend, ExecuteSupport, etc.)
    │   ├─ Evaluators (SyncCommand, UpdateObservation)
    │   └─ Conditions (CheckCommandType, CheckTacticalAction, IsAlive)
    ├─ Combat Systems (Health, Weapon)
    └─ Reward Calculator (hierarchical rewards)
```

---

## Architecture Components

### Core Files

#### Strategic Layer
- **Team/TeamLeaderComponent.h/cpp** (TeamLeaderComponent.cpp:1-800+)
  - Continuous MCTS execution (1.5s configurable intervals)
  - Strategic command generation with uncertainty
  - Curriculum scenario tracking
  - Performance profiling

- **AI/MCTS/MCTS.h/cpp** (MCTS.cpp:1-600+)
  - AlphaZero-style tree search with PUCT
  - Value network integration for leaf evaluation
  - World model integration for multi-step rollouts
  - Prior-guided node expansion
  - UCB-based action sampling

- **AI/MCTS/TeamMCTSNode.h** (TeamMCTSNode.h:1-120)
  - Node structure with ActionPriors
  - PUCT calculation for exploration-exploitation balance
  - Visit count and value tracking

#### Tactical Layer
- **Team/FollowerAgentComponent.h/cpp** (FollowerAgentComponent.cpp:1-900+)
  - Observation building (71 individual + 40 team features)
  - RL policy integration
  - Combat event handling
  - Confidence-weighted command execution
  - State transition logging

- **RL/RLPolicyNetwork.h/cpp** (RLPolicyNetwork.h:1-200, RLPolicyNetwork.cpp:1-500+)
  - 3-layer MLP (128→128→64 neurons)
  - 16 tactical actions (Move, Assault, Defend, TakeCover, etc.)
  - Objective-based priors for MCTS
  - Experience storage with MCTS uncertainty tagging
  - ONNX inference via NNE

#### Neural Networks
- **RL/TeamValueNetwork.h/cpp** (TeamValueNetwork.cpp:1-400+)
  - Input: 40 team + N×71 individual features
  - Architecture: Embedding(256) → Trunk(256→256→128) → Value(128→64→1)
  - Output: Team state value [-1, 1] (loss → win probability)
  - Training: TD-learning on MCTS outcomes

- **Simulation/WorldModel.h/cpp** (WorldModel.cpp:1-500+)
  - Input: TeamObservation + All actions (strategic + tactical)
  - Architecture: ActionEncoder → StateEncoder → Fusion → Transition Predictor
  - Output: Predicted next state (FStateTransition)
  - Training: Supervised learning on real transitions
  - Integration: 5-step rollouts in MCTS simulation

- **RL/HybridPolicyNetwork.h/cpp** (HybridPolicyNetwork.cpp:1-300)
  - Dual-head architecture
  - Policy Head: Softmax probabilities (immediate action selection)
  - Prior Head: Logits (MCTS node initialization)
  - Training: PPO with MCTS prior targets

#### Training Support
- **RL/CurriculumManager.h/cpp** (CurriculumManager.cpp:1-300)
  - Tracks MCTS high-uncertainty scenarios
  - Prioritizes hard situations for RL training
  - Configurable sampling strategies

- **RL/RewardCalculator.h/cpp** (RewardCalculator.cpp:1-400+)
  - Hierarchical reward computation
  - Individual: +10 kill, +5 damage, -5 take damage, -10 death
  - Coordination: +15 strategic kill, +10 combined fire, +5 formation
  - Strategic: +50 objective, +30 team wipe
  - Tracks formation adherence, objective compliance

---

## Team Leader System (Strategic MCTS)

### Continuous Planning

**File:** TeamLeaderComponent.cpp:200-250

```cpp
// Continuous planning (1.5s intervals, configurable)
void UTeamLeaderComponent::BeginPlay()
{
    // Start continuous planning timer
    GetWorld()->GetTimerManager().SetTimer(
        ContinuousPlanningTimerHandle,
        this,
        &UTeamLeaderComponent::RunContinuousMCTS,
        ContinuousPlanningInterval,  // Default: 1.5s
        true  // Loop
    );
}

void UTeamLeaderComponent::RunContinuousMCTS()
{
    if (!bEnableContinuousPlanning) return;

    // Can be interrupted by critical events (priority ≥9)
    if (HasCriticalEvent())
    {
        UE_LOG(LogTemp, Warning, TEXT("Critical event interrupting continuous planning"));
        RunEventDrivenMCTS();
        return;
    }

    // Run proactive MCTS
    RunMCTSAsync(MaxSimulations, SearchDepth);
}
```

**Key Features:**
- Proactive planning every 1-2s (configurable)
- Critical events can interrupt (priority threshold)
- Async execution on background thread
- Performance profiling with rolling averages

### MCTS Tree Search

**File:** MCTS.cpp:100-400

#### Selection (PUCT with Priors)

```cpp
// AlphaZero-style PUCT calculation
float FTeamMCTSNode::CalculatePUCT(int32 ParentVisitCount, float ExplorationConstant) const
{
    if (VisitCount == 0) return FLT_MAX;  // Unvisited nodes prioritized

    float exploitation = Value / VisitCount;

    // Prior probability (from RL policy)
    float prior = Prior > 0.0f ? Prior : (1.0f / ParentChildCount);

    // PUCT formula: Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    float exploration = ExplorationConstant * prior *
                       FMath::Sqrt(ParentVisitCount) / (1.0f + VisitCount);

    return exploitation + exploration;
}
```

#### Expansion (Prior-Guided)

**File:** MCTS.cpp:250-300

```cpp
void UMCTS::ExpandNode(TSharedPtr<FTeamMCTSNode> Node)
{
    // Generate command combinations using UCB
    TArray<TMap<AActor*, FStrategicCommand>> CommandSets =
        GenerateCommandCombinationsUCB(Node, Followers, MaxCombinations);

    // Query RL policy for priors
    TArray<float> Priors = RLPolicyNetwork->GetObjectivePriors(
        Node->TeamObservation, Followers, CommandSets);

    // Create child nodes with priors
    for (int32 i = 0; i < CommandSets.Num(); ++i)
    {
        TSharedPtr<FTeamMCTSNode> Child = MakeShared<FTeamMCTSNode>();
        Child->Commands = CommandSets[i];
        Child->Prior = Priors[i];  // Initialize with RL prior

        Node->Children.Add(Child);
        Node->UntriedActions.Add(CommandSets[i]);
        Node->ActionPriors.Add(Priors[i]);
    }
}
```

**UCB Action Sampling:**
- Top-3 objectives per follower (based on distance, threat, strategic value)
- Synergy bonuses for complementary actions
- Epsilon-greedy exploration (20%)
- Progressive widening as visit count increases

#### Simulation (World Model Rollouts)

**File:** MCTS.cpp:350-400

```cpp
float UMCTS::SimulateNode(TSharedPtr<FTeamMCTSNode> Node, int32 Depth)
{
    // Base case: max depth or terminal state
    if (Depth >= MaxDepth || Node->IsTerminal())
    {
        // Use Value Network for leaf evaluation
        return ValueNetwork->EvaluateTeamState(Node->TeamObservation);
    }

    // World Model: Multi-step rollout (5 steps)
    FTeamObservation CurrentState = Node->TeamObservation;
    float AccumulatedReward = 0.0f;
    float DiscountFactor = 0.95f;

    for (int32 Step = 0; Step < 5; ++Step)
    {
        // Predict next state
        FStateTransition Transition = WorldModel->PredictTransition(
            CurrentState, Node->Commands, Followers);

        // Apply predicted changes
        CurrentState.ApplyDelta(Transition);

        // Accumulate reward
        float StepReward = Transition.Reward;
        AccumulatedReward += FMath::Pow(DiscountFactor, Step) * StepReward;

        // Check for terminal state
        if (Transition.bIsTerminal)
            break;
    }

    // Final state evaluation
    float FinalValue = ValueNetwork->EvaluateTeamState(CurrentState);

    return AccumulatedReward + FMath::Pow(DiscountFactor, 5) * FinalValue;
}
```

#### Backpropagation

**File:** MCTS.cpp:420-450

```cpp
void UMCTS::BackpropagateNode(TSharedPtr<FTeamMCTSNode> Node, float Value)
{
    TSharedPtr<FTeamMCTSNode> Current = Node;

    while (Current.IsValid())
    {
        Current->VisitCount++;
        Current->Value += Value;  // Accumulate value

        // Move to parent
        Current = Current->Parent.Pin();

        // Alternate perspective (minimax for enemy turns)
        Value = -Value;
    }
}
```

### Command Generation

**File:** TeamLeaderComponent.cpp:400-500

```cpp
void UTeamLeaderComponent::IssueStrategicCommands()
{
    if (!MCTSRoot || MCTSRoot->Children.Num() == 0)
        return;

    // Select best child (highest visit count)
    TSharedPtr<FTeamMCTSNode> BestChild = SelectBestChild(MCTSRoot);

    // Compute uncertainty metrics
    float Confidence = (float)BestChild->VisitCount / (float)MCTSRoot->VisitCount;
    float ValueVariance = ComputeValueVariance(MCTSRoot);
    float PolicyEntropy = ComputePolicyEntropy(MCTSRoot);

    // Issue commands to followers
    for (const auto& Pair : BestChild->Commands)
    {
        AActor* Follower = Pair.Key;
        FStrategicCommand Command = Pair.Value;

        // Add uncertainty quantification
        Command.Confidence = Confidence;
        Command.ValueVariance = ValueVariance;
        Command.PolicyEntropy = PolicyEntropy;

        SendCommandToFollower(Follower, Command);
    }

    // Log for curriculum manager
    if (Confidence < 0.5f || PolicyEntropy > 1.5f)
    {
        CurriculumManager->RecordHighUncertaintyScenario(
            MCTSRoot->TeamObservation, BestChild->Commands,
            Confidence, ValueVariance, PolicyEntropy);
    }
}
```

### Uncertainty Quantification

**File:** TeamLeaderComponent.cpp:550-600

```cpp
float UTeamLeaderComponent::ComputeValueVariance(TSharedPtr<FTeamMCTSNode> Node)
{
    if (Node->Children.Num() == 0)
        return 0.0f;

    // Compute mean value
    float MeanValue = 0.0f;
    int32 TotalVisits = 0;
    for (const auto& Child : Node->Children)
    {
        if (Child->VisitCount > 0)
        {
            float ChildValue = Child->Value / Child->VisitCount;
            MeanValue += ChildValue * Child->VisitCount;
            TotalVisits += Child->VisitCount;
        }
    }
    MeanValue /= FMath::Max(TotalVisits, 1);

    // Compute variance
    float Variance = 0.0f;
    for (const auto& Child : Node->Children)
    {
        if (Child->VisitCount > 0)
        {
            float ChildValue = Child->Value / Child->VisitCount;
            float Diff = ChildValue - MeanValue;
            Variance += Diff * Diff * Child->VisitCount;
        }
    }
    Variance /= FMath::Max(TotalVisits, 1);

    return FMath::Sqrt(Variance);
}

float UTeamLeaderComponent::ComputePolicyEntropy(TSharedPtr<FTeamMCTSNode> Node)
{
    // Shannon entropy: H(π) = -Σ π(a) log π(a)
    float Entropy = 0.0f;
    int32 TotalVisits = Node->VisitCount;

    for (const auto& Child : Node->Children)
    {
        if (Child->VisitCount > 0 && TotalVisits > 0)
        {
            float Probability = (float)Child->VisitCount / (float)TotalVisits;
            Entropy -= Probability * FMath::Loge(Probability);
        }
    }

    return Entropy;
}
```

---

## Follower Agent System (Tactical RL)

### RL Policy Network Integration

**File:** FollowerAgentComponent.cpp:200-300

```cpp
void UFollowerAgentComponent::SelectTacticalAction()
{
    // Build observation
    FObservationElement Observation = BuildObservation();

    // Query RL policy
    ETacticalAction Action = RLPolicyNetwork->SelectAction(Observation);

    // Set action in StateTree context
    FollowerStateTreeContext.CurrentTacticalAction = Action;

    // Update StateTree (will transition states based on action)
    StateTreeComponent->Tick(DeltaTime);
}
```

### Confidence-Weighted Command Execution

**File:** FollowerAgentComponent.cpp:350-400

```cpp
void UFollowerAgentComponent::OnReceiveStrategicCommand(const FStrategicCommand& Command)
{
    CurrentStrategicCommand = Command;

    // Check confidence threshold
    if (Command.Confidence < ConfidenceThreshold)  // Default: 0.5
    {
        UE_LOG(LogAI, Warning, TEXT("%s: Low confidence command (%.2f), RL may override"),
               *GetName(), Command.Confidence);

        bAllowTacticalOverride = true;
    }
    else
    {
        bAllowTacticalOverride = false;
    }

    // Log for debugging
    if (Command.PolicyEntropy > 1.5f)
    {
        UE_LOG(LogAI, Warning, TEXT("%s: High entropy command (%.2f), ambiguous situation"),
               *GetName(), Command.PolicyEntropy);
    }

    // Sync with StateTree
    FollowerStateTreeContext.CurrentStrategicCommand = Command;
}
```

### StateTree Execution

**File:** StateTree/FollowerStateTreeComponent.cpp:50-150

StateTree is the **PRIMARY** execution system. It:
- Drives all state transitions (Idle, Assault, Defend, Support, Move, Retreat)
- Executes tactical actions via Tasks
- Syncs with strategic commands via Evaluators
- Checks conditions for state transitions

**Key Tasks:**
- `STTask_QueryRLPolicy` - Queries RL network for tactical action
- `STTask_ExecuteAssault` - Assault state execution (weapon firing, movement)
- `STTask_ExecuteDefend` - Defend state execution (cover, overwatch)
- `STTask_ExecuteSupport` - Support state execution (healing, covering allies)

**Key Evaluators:**
- `STEvaluator_SyncCommand` - Syncs strategic commands from leader
- `STEvaluator_UpdateObservation` - Updates observation data

**Key Conditions:**
- `STCondition_CheckCommandType` - Checks if command type matches
- `STCondition_CheckTacticalAction` - Checks if tactical action matches
- `STCondition_IsAlive` - Checks if agent is alive

### Experience Storage

**File:** FollowerAgentComponent.cpp:700-750

```cpp
void UFollowerAgentComponent::StoreExperience()
{
    if (!RLPolicyNetwork) return;

    // Get MCTS uncertainty metrics (if available)
    float MCTSConfidence = CurrentStrategicCommand.Confidence;
    float MCTSVariance = CurrentStrategicCommand.ValueVariance;
    float MCTSEntropy = CurrentStrategicCommand.PolicyEntropy;

    // Store experience with MCTS tagging
    RLPolicyNetwork->StoreExperienceWithUncertainty(
        PreviousObservation,
        PreviousTacticalAction,
        AccumulatedReward,
        CurrentObservation,
        bIsDone,
        MCTSConfidence,
        MCTSVariance,
        MCTSEntropy
    );

    // Reset accumulator
    AccumulatedReward = 0.0f;
}
```

---

## Value Network

**File:** RL/TeamValueNetwork.cpp:1-400

### Architecture

```
Input: FTeamObservation (40 team + N×71 individual features)
  ↓
Embedding Layer (Linear: InputDim → 256, ReLU)
  ↓
Shared Trunk
  ├─ Layer 1: Linear(256 → 256, ReLU)
  ├─ Layer 2: Linear(256 → 128, ReLU)
  ↓
Value Head
  ├─ Layer 1: Linear(128 → 64, ReLU)
  └─ Layer 2: Linear(64 → 1, Tanh)
  ↓
Output: Team state value [-1, 1] (loss → win probability)
```

### Inference (C++)

```cpp
float UTeamValueNetwork::EvaluateTeamState(const FTeamObservation& TeamObs)
{
    if (!NNEModel)
    {
        UE_LOG(LogAI, Error, TEXT("ValueNetwork: No ONNX model loaded"));
        return 0.0f;  // Neutral value
    }

    // Flatten observation to input tensor
    TArray<float> InputTensor = TeamObs.Flatten();

    // Run inference via NNE
    TArray<float> OutputTensor;
    bool bSuccess = NNEModel->Evaluate(InputTensor, OutputTensor);

    if (!bSuccess || OutputTensor.Num() == 0)
    {
        UE_LOG(LogAI, Error, TEXT("ValueNetwork: Inference failed"));
        return 0.0f;
    }

    // Output is in [-1, 1] via Tanh activation
    float Value = OutputTensor[0];

    return FMath::Clamp(Value, -1.0f, 1.0f);
}
```

### Training (Python)

**File:** Scripts/train_value_network.py:1-300

```python
# TD-Learning on MCTS rollout outcomes
def train_value_network(model, data, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for batch in data_loader:
            team_obs, final_outcome = batch

            # Forward pass
            predicted_value = model(team_obs)

            # Loss: MSE between predicted and actual outcome
            loss = criterion(predicted_value, final_outcome)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model
```

---

## World Model

**File:** Simulation/WorldModel.cpp:1-500

### Architecture

```
Input: CurrentState (TeamObs) + AllActions (strategic + tactical)
  ↓
Action Encoder (commands → embeddings, 64-dim per action)
  ↓
State Encoder (observations → embeddings, 256-dim)
  ↓
Fusion Layer (Concatenate + MLP: 320 → 256 → 256)
  ↓
Transition Predictor
  ├─ Health Delta Head (256 → 128 → N_agents)
  ├─ Position Delta Head (256 → 128 → N_agents×3)
  ├─ Status Effect Head (256 → 128 → N_agents×StatusDim)
  └─ Terminal Prediction (256 → 64 → 1, Sigmoid)
  ↓
Output: FStateTransition (deltas + terminal flag + reward)
```

### Inference (C++)

```cpp
FStateTransition UWorldModel::PredictTransition(
    const FTeamObservation& CurrentState,
    const TMap<AActor*, FStrategicCommand>& Commands,
    const TArray<AActor*>& Followers)
{
    FStateTransition Transition;

    if (!NNEModel)
    {
        // Fallback: Use heuristic prediction
        return PredictTransitionHeuristic(CurrentState, Commands);
    }

    // Flatten inputs
    TArray<float> StateInput = CurrentState.Flatten();
    TArray<float> ActionInput = FlattenActions(Commands, Followers);

    // Concatenate
    TArray<float> CombinedInput;
    CombinedInput.Append(StateInput);
    CombinedInput.Append(ActionInput);

    // Run inference
    TArray<float> OutputTensor;
    bool bSuccess = NNEModel->Evaluate(CombinedInput, OutputTensor);

    if (!bSuccess)
    {
        return PredictTransitionHeuristic(CurrentState, Commands);
    }

    // Parse output tensor
    int32 Offset = 0;
    int32 NumAgents = Followers.Num();

    // Health deltas
    for (int32 i = 0; i < NumAgents; ++i)
    {
        Transition.HealthDeltas.Add(OutputTensor[Offset++]);
    }

    // Position deltas
    for (int32 i = 0; i < NumAgents; ++i)
    {
        FVector Delta(
            OutputTensor[Offset++],
            OutputTensor[Offset++],
            OutputTensor[Offset++]
        );
        Transition.PositionDeltas.Add(Delta);
    }

    // Terminal flag
    Transition.bIsTerminal = OutputTensor[Offset++] > 0.5f;

    // Estimated reward (optional)
    Transition.Reward = OutputTensor[Offset++];

    return Transition;
}
```

### Training (Python)

**File:** Scripts/train_world_model.py:1-400

```python
# Supervised learning on real transitions
def train_world_model(model, transitions, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    for epoch in range(epochs):
        for batch in data_loader:
            current_state, actions, next_state, is_terminal = batch

            # Forward pass
            predicted_next_state, terminal_prob = model(current_state, actions)

            # Loss: MSE on state prediction + BCE on terminal flag
            state_loss = mse_loss(predicted_next_state, next_state)
            terminal_loss = bce_loss(terminal_prob, is_terminal)

            loss = state_loss + 0.1 * terminal_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model
```

---

## Coupled Training

### MCTS → RL: Curriculum Manager

**File:** RL/CurriculumManager.cpp:1-300

```cpp
void UCurriculumManager::RecordHighUncertaintyScenario(
    const FTeamObservation& TeamObs,
    const TMap<AActor*, FStrategicCommand>& Commands,
    float Confidence, float ValueVariance, float PolicyEntropy)
{
    // Compute uncertainty score
    float UncertaintyScore = (1.0f - Confidence) + ValueVariance + PolicyEntropy;

    // Store scenario
    FMCTSScenario Scenario;
    Scenario.TeamObservation = TeamObs;
    Scenario.Commands = Commands;
    Scenario.UncertaintyScore = UncertaintyScore;
    Scenario.Timestamp = FDateTime::Now();

    HighUncertaintyScenarios.Add(Scenario);

    // Maintain buffer size
    if (HighUncertaintyScenarios.Num() > MaxBufferSize)
    {
        // Remove oldest or lowest uncertainty
        HighUncertaintyScenarios.RemoveAt(0);
    }

    UE_LOG(LogAI, Log, TEXT("CurriculumManager: Recorded scenario (uncertainty: %.2f)"),
           UncertaintyScore);
}

TArray<FMCTSScenario> UCurriculumManager::SampleHardScenarios(int32 NumSamples)
{
    // Sort by uncertainty score (descending)
    HighUncertaintyScenarios.Sort([](const FMCTSScenario& A, const FMCTSScenario& B) {
        return A.UncertaintyScore > B.UncertaintyScore;
    });

    // Take top-N
    TArray<FMCTSScenario> Sampled;
    for (int32 i = 0; i < FMath::Min(NumSamples, HighUncertaintyScenarios.Num()); ++i)
    {
        Sampled.Add(HighUncertaintyScenarios[i]);
    }

    return Sampled;
}
```

### RL → MCTS: Policy Priors

**File:** RL/RLPolicyNetwork.cpp:400-500

```cpp
TArray<float> URLPolicyNetwork::GetObjectivePriors(
    const FTeamObservation& TeamObs,
    const TArray<AActor*>& Followers,
    const TArray<TMap<AActor*, FStrategicCommand>>& CommandSets)
{
    TArray<float> Priors;
    Priors.Reserve(CommandSets.Num());

    for (const auto& CommandSet : CommandSets)
    {
        float CommandSetPrior = 1.0f;

        // Compute prior for each follower's command
        for (const auto& Pair : CommandSet)
        {
            AActor* Follower = Pair.Key;
            FStrategicCommand Command = Pair.Value;

            // Context-aware prior (heuristic-based for now)
            float CommandPrior = ComputeCommandPrior(Follower, Command, TeamObs);

            // Multiply priors (joint probability)
            CommandSetPrior *= CommandPrior;
        }

        Priors.Add(CommandSetPrior);
    }

    // Normalize
    float Sum = 0.0f;
    for (float Prior : Priors)
        Sum += Prior;

    if (Sum > 0.0f)
    {
        for (float& Prior : Priors)
            Prior /= Sum;
    }
    else
    {
        // Uniform if all zeros
        float Uniform = 1.0f / FMath::Max(Priors.Num(), 1);
        for (float& Prior : Priors)
            Prior = Uniform;
    }

    return Priors;
}
```

---

## Reward System

**File:** RL/RewardCalculator.cpp:1-400

### Hierarchical Reward Structure

```cpp
float URewardCalculator::CalculateReward(
    const FRewardContext& Context)
{
    float TotalReward = 0.0f;

    // === Individual Rewards ===
    TotalReward += Context.Kills * IndividualReward_Kill;          // +10
    TotalReward += Context.DamageDealt * IndividualReward_Damage;  // +5
    TotalReward += Context.DamageTaken * IndividualReward_TakeDamage; // -5
    if (Context.bDied)
        TotalReward += IndividualReward_Death;                     // -10

    // === Coordination Bonuses ===
    if (Context.bExecutingStrategicCommand)
    {
        if (Context.Kills > 0)
            TotalReward += CoordinationBonus_StrategicKill;        // +15
    }

    if (Context.bCombinedFire)
        TotalReward += CoordinationBonus_CombinedFire;             // +10

    if (Context.bInFormation)
        TotalReward += CoordinationBonus_Formation;                // +5

    if (Context.bDisobeyedCommand)
        TotalReward += CoordinationPenalty_Disobey;                // -15

    // === Strategic Rewards (Team-level, propagated to all) ===
    if (Context.bObjectiveCaptured)
        TotalReward += StrategicReward_Objective;                  // +50

    if (Context.bEnemyTeamWiped)
        TotalReward += StrategicReward_EnemyWipe;                  // +30

    if (Context.bOwnTeamWiped)
        TotalReward += StrategicPenalty_OwnWipe;                   // -30

    return TotalReward;
}
```

### Coordination Detection

```cpp
bool URewardCalculator::DetectCombinedFire(
    AActor* Agent,
    AActor* Target,
    const TArray<AActor*>& Allies)
{
    // Check if multiple allies are targeting the same enemy
    int32 AlliesTargetingSame = 0;
    float CombinedFireWindow = 2.0f;  // 2 second window

    for (AActor* Ally : Allies)
    {
        if (Ally == Agent) continue;

        // Check if ally attacked same target recently
        if (IsTargetingEnemy(Ally, Target, CombinedFireWindow))
        {
            AlliesTargetingSame++;
        }
    }

    return AlliesTargetingSame >= 1;  // At least one ally attacking same target
}

bool URewardCalculator::DetectFormation(
    AActor* Agent,
    const TArray<AActor*>& Allies,
    const FStrategicCommand& Command)
{
    // Check if agent is maintaining formation with allies
    FVector AgentPos = Agent->GetActorLocation();
    FVector TargetPos = Command.TargetLocation;

    // Expected position based on command
    FVector ExpectedPos = CalculateExpectedPosition(Agent, Command, Allies);

    // Distance tolerance
    float DistanceTolerance = 500.0f;  // 5 meters
    float Distance = FVector::Dist(AgentPos, ExpectedPos);

    return Distance < DistanceTolerance;
}
```

---

## Observation System

**File:** Observation/ObservationElement.cpp:1-400, TeamObservation.cpp:1-300

### Individual Observation (71 features)

```cpp
FObservationElement UFollowerAgentComponent::BuildObservation()
{
    FObservationElement Obs;

    // === Self State (10) ===
    Obs.SelfHealth = HealthComponent->GetHealthNormalized();
    Obs.SelfArmor = HealthComponent->GetArmorNormalized();
    Obs.SelfPosition = GetActorLocation() / 10000.0f;  // Normalize
    Obs.SelfRotation = GetActorRotation().Vector();
    Obs.SelfVelocity = GetVelocity() / 600.0f;  // Normalize by max speed
    Obs.SelfAmmo = WeaponComponent->GetAmmoNormalized();
    Obs.SelfCooldown = WeaponComponent->GetCooldownNormalized();

    // === Command State (5) ===
    Obs.CommandType = (float)CurrentStrategicCommand.CommandType / 5.0f;
    Obs.CommandTargetLocation = CurrentStrategicCommand.TargetLocation / 10000.0f;
    Obs.CommandConfidence = CurrentStrategicCommand.Confidence;

    // === Enemy State (30) - Top 3 enemies ===
    TArray<AActor*> NearbyEnemies = PerceptionComponent->GetPerceivedEnemies();
    for (int32 i = 0; i < FMath::Min(3, NearbyEnemies.Num()); ++i)
    {
        AActor* Enemy = NearbyEnemies[i];
        Obs.EnemyRelativePosition[i] = (Enemy->GetActorLocation() - GetActorLocation()) / 10000.0f;
        Obs.EnemyDistance[i] = FVector::Dist(GetActorLocation(), Enemy->GetActorLocation()) / 10000.0f;
        Obs.EnemyHealth[i] = GetEnemyHealth(Enemy);
    }

    // === Ally State (20) - Top 2 allies ===
    TArray<AActor*> Allies = TeamLeader->GetFollowers();
    for (int32 i = 0; i < FMath::Min(2, Allies.Num()); ++i)
    {
        if (Allies[i] == this) continue;
        Obs.AllyRelativePosition[i] = (Allies[i]->GetActorLocation() - GetActorLocation()) / 10000.0f;
        Obs.AllyHealth[i] = GetAllyHealth(Allies[i]);
    }

    // === Environment (6) - Raycast to cover ===
    for (int32 i = 0; i < 6; ++i)
    {
        float Angle = i * 60.0f;  // 360° / 6
        FVector Direction = GetActorRotation().RotateVector(FVector::ForwardVector);
        Direction = Direction.RotateAngleAxis(Angle, FVector::UpVector);

        FHitResult Hit;
        bool bHit = GetWorld()->LineTraceSingleByChannel(
            Hit, GetActorLocation(), GetActorLocation() + Direction * 1000.0f,
            ECC_Visibility);

        Obs.EnvironmentDistances[i] = bHit ? (Hit.Distance / 1000.0f) : 1.0f;
    }

    return Obs;
}
```

### Team Observation (40 features)

```cpp
FTeamObservation UTeamLeaderComponent::BuildTeamObservation()
{
    FTeamObservation TeamObs;

    // === Team Aggregate State (15) ===
    TeamObs.TeamAvgHealth = ComputeAverageHealth(Followers);
    TeamObs.TeamAvgAmmo = ComputeAverageAmmo(Followers);
    TeamObs.TeamCentroid = ComputeCentroid(Followers) / 10000.0f;
    TeamObs.TeamSpread = ComputeSpread(Followers) / 10000.0f;
    TeamObs.TeamFormationQuality = ComputeFormationQuality(Followers);

    // === Objective State (10) ===
    TeamObs.ObjectiveDistance = FVector::Dist(TeamObs.TeamCentroid, ObjectiveLocation) / 10000.0f;
    TeamObs.ObjectiveOwnership = CurrentObjectiveOwnership;  // -1, 0, 1
    TeamObs.ObjectiveCaptureProgress = CaptureProgress;

    // === Enemy Threat Assessment (15) ===
    TArray<AActor*> AllEnemies = GatherAllPerceivedEnemies(Followers);
    TeamObs.EnemyCount = AllEnemies.Num() / 10.0f;  // Normalize
    TeamObs.EnemyAvgDistance = ComputeAverageDistance(TeamObs.TeamCentroid, AllEnemies) / 10000.0f;
    TeamObs.EnemyThreatLevel = AssessThreatLevel(AllEnemies, Followers);

    // === Individual Observations (N × 71) ===
    for (AActor* Follower : Followers)
    {
        FObservationElement IndividualObs = CastChecked<UFollowerAgentComponent>(
            Follower->GetComponentByClass(UFollowerAgentComponent::StaticClass())
        )->BuildObservation();

        TeamObs.IndividualObservations.Add(IndividualObs);
    }

    return TeamObs;
}
```

---

## Training Pipeline

### Self-Play Data Collection

**Script:** Scripts/self_play_collector.py

```bash
# Collect 1000 games of self-play data
python self_play_collector.py --games 1000 --output ./selfplay_data --save-interval 50

# Data collected:
# - RL experiences (observations, actions, rewards, next_obs)
# - MCTS traces (team_obs, commands, visit_counts, final_outcome)
# - State transitions (current_state, actions, next_state)
# - Game outcomes (win/loss/draw, kills, deaths, coordination metrics)
```

### Coupled Training

**Script:** Scripts/train_coupled_system.py

```bash
# Train all networks on collected data
python train_coupled_system.py \
    --data-dir ./selfplay_data \
    --output-dir ./training_output \
    --iterations 1 \
    --value-epochs 50 \
    --world-epochs 50 \
    --rl-epochs 50 \
    --batch-size 64 \
    --copy-to-ue5 ../Content/AI/Models

# Trains:
# 1. ValueNetwork on MCTS outcomes (TD-learning)
# 2. WorldModel on state transitions (supervised)
# 3. RLPolicy on RL experiences (PPO with prioritized replay)
#
# Exports to ONNX and copies to UE5 project
```

### Evaluation

**Script:** Scripts/evaluate_agents.py

```bash
# Compare v3.0 against v2.0 baseline
python evaluate_agents.py \
    --data ./evaluation_data \
    --output ./evaluation_results \
    --baseline v2.0 \
    --trained v3.0 \
    --plots

# Metrics:
# - Win rate (target: ≥70%)
# - K/D ratio
# - Coordination rate (target: ≥30%)
# - MCTS efficiency (target: 50% fewer simulations)
# - Damage efficiency
```

### Complete Pipeline

**Script:** Scripts/run_selfplay_pipeline.py

```bash
# Run complete training loop (1000 games over 10 iterations)
python run_selfplay_pipeline.py \
    --games 100 \
    --iterations 10 \
    --output ./pipeline_output \
    --ue5-project "C:/Projects/SBDAPM"

# Pipeline steps (per iteration):
# 1. Collect 100 games of self-play data
# 2. Train all networks
# 3. Export to ONNX and deploy to UE5
# 4. Evaluate (every 2 iterations)
# 5. Repeat
```

---

## Performance Characteristics

### Timing Targets (v3.0)

| Component | Target | Current (v2.0) | Notes |
|-----------|--------|----------------|-------|
| MCTS Tree Search | 30-50ms | ~34ms | Improved with value network pruning |
| RL Inference | 1-3ms | ~2ms | NNE optimized, GPU inference |
| World Model Prediction | 5-10ms | N/A | 5-step rollout |
| Value Network Inference | 2-5ms | N/A | Single forward pass |
| StateTree Tick | <0.5ms | ~0.3ms | Per agent |
| **Total Frame Budget** | **10-20ms** | **~15ms** | For 4-agent team |

### Memory Usage

- MCTS Tree: ~5-10 MB (1000 simulations, 10 depth)
- Value Network: ~2 MB (ONNX model)
- World Model: ~5 MB (ONNX model)
- RL Policy: ~1 MB (ONNX model)
- Observations: ~50 KB per agent

### Scalability

- Tested with **4-8 agents per team**
- MCTS scales linearly with team size
- RL inference per agent (independent)
- World model scales with state size (quadratic in agent count)

---

## Configuration

### MCTS Parameters

**File:** AI/MCTS/MCTS.h:30-50

```cpp
UPROPERTY(EditAnywhere, Category = "MCTS")
int32 MaxSimulations = 1000;  // Simulations per MCTS run

UPROPERTY(EditAnywhere, Category = "MCTS")
int32 SearchDepth = 10;  // Max tree depth

UPROPERTY(EditAnywhere, Category = "MCTS")
float ExplorationConstant = 1.41f;  // UCB exploration (sqrt(2))

UPROPERTY(EditAnywhere, Category = "MCTS")
int32 MaxCombinations = 10;  // Action combinations to expand

UPROPERTY(EditAnywhere, Category = "MCTS")
float EpsilonGreedy = 0.2f;  // Exploration probability
```

### Continuous Planning

**File:** Team/TeamLeaderComponent.h:80-90

```cpp
UPROPERTY(EditAnywhere, Category = "Continuous Planning")
bool bEnableContinuousPlanning = true;

UPROPERTY(EditAnywhere, Category = "Continuous Planning")
float ContinuousPlanningInterval = 1.5f;  // Seconds between MCTS runs

UPROPERTY(EditAnywhere, Category = "Continuous Planning")
int32 CriticalEventPriority = 9;  // Priority threshold to interrupt planning
```

### Confidence Thresholds

**File:** Team/FollowerAgentComponent.h:100-110

```cpp
UPROPERTY(EditAnywhere, Category = "Command Execution")
float ConfidenceThreshold = 0.5f;  // Minimum confidence to execute command

UPROPERTY(EditAnywhere, Category = "Command Execution")
float HighEntropyThreshold = 1.5f;  // Threshold for high uncertainty warning

UPROPERTY(EditAnywhere, Category = "Command Execution")
bool bAllowTacticalOverride = true;  // Allow RL to override low-confidence commands
```

### Reward Weights

**File:** RL/RewardCalculator.h:40-70

```cpp
// Individual rewards
UPROPERTY(EditAnywhere, Category = "Rewards|Individual")
float IndividualReward_Kill = 10.0f;

UPROPERTY(EditAnywhere, Category = "Rewards|Individual")
float IndividualReward_Damage = 5.0f;

UPROPERTY(EditAnywhere, Category = "Rewards|Individual")
float IndividualReward_TakeDamage = -5.0f;

UPROPERTY(EditAnywhere, Category = "Rewards|Individual")
float IndividualReward_Death = -10.0f;

// Coordination bonuses
UPROPERTY(EditAnywhere, Category = "Rewards|Coordination")
float CoordinationBonus_StrategicKill = 15.0f;

UPROPERTY(EditAnywhere, Category = "Rewards|Coordination")
float CoordinationBonus_CombinedFire = 10.0f;

UPROPERTY(EditAnywhere, Category = "Rewards|Coordination")
float CoordinationBonus_Formation = 5.0f;

UPROPERTY(EditAnywhere, Category = "Rewards|Coordination")
float CoordinationPenalty_Disobey = -15.0f;

// Strategic rewards
UPROPERTY(EditAnywhere, Category = "Rewards|Strategic")
float StrategicReward_Objective = 50.0f;

UPROPERTY(EditAnywhere, Category = "Rewards|Strategic")
float StrategicReward_EnemyWipe = 30.0f;

UPROPERTY(EditAnywhere, Category = "Rewards|Strategic")
float StrategicPenalty_OwnWipe = -30.0f;
```

---

## Troubleshooting

### MCTS Performance Issues

**Symptom:** MCTS taking >100ms

**Causes:**
1. Too many simulations (MaxSimulations > 1500)
2. Too many action combinations (MaxCombinations > 20)
3. Deep search depth (SearchDepth > 15)
4. Value network inference slow

**Solutions:**
- Reduce MaxSimulations to 500-1000
- Limit MaxCombinations to 10-15
- Reduce SearchDepth to 8-10
- Ensure ONNX models are optimized (quantized)
- Check GPU inference is enabled (NNE settings)

### Value Network Not Loading

**Symptom:** "ValueNetwork: No ONNX model loaded" warnings

**Causes:**
1. ONNX file not in correct directory
2. Model name mismatch
3. NNE plugin not enabled

**Solutions:**
- Verify ONNX file in `Content/AI/Models/value_network_latest.onnx`
- Check model path in TeamLeaderComponent properties
- Enable NNE plugin in Project Settings

### Low Confidence Commands

**Symptom:** Frequent "Low confidence command" warnings

**Causes:**
1. Insufficient MCTS simulations
2. High policy entropy (ambiguous situations)
3. Value network not trained

**Solutions:**
- Increase MaxSimulations
- Train value network on more data
- Lower ConfidenceThreshold (but may reduce quality)
- Check world model predictions are accurate

### Coordination Not Detected

**Symptom:** Low coordination bonus despite visual coordination

**Causes:**
1. CombinedFireWindow too narrow
2. Formation tolerance too strict
3. Timing issues (actions not synchronized)

**Solutions:**
- Increase CombinedFireWindow from 2.0s to 3.0s
- Increase DistanceTolerance from 500 to 800
- Check command sync timing in TeamCommunicationManager

### Training Not Converging

**Symptom:** Loss plateaus, no improvement

**Causes:**
1. Learning rate too high/low
2. Insufficient data diversity
3. Reward scaling issues
4. Overfitting

**Solutions:**
- Adjust learning rates (0.0001-0.001 for value/world, 0.0003 for RL)
- Collect more diverse self-play data
- Normalize rewards (divide by max episode reward)
- Add dropout (0.2-0.3) and L2 regularization

---

## Next Steps

1. **Data Collection**: Run 1000+ self-play games in UE5 with data export enabled
2. **Training**: Execute `python run_selfplay_pipeline.py --games 1000 --iterations 10`
3. **Evaluation**: Compare v3.0 against v2.0 baseline
4. **Iteration**: Repeat training cycle based on evaluation results
5. **Deployment**: Use trained models for production gameplay

---

**Document Version:** 1.0
**Last Updated:** 2025-11-26
**Maintained By:** SBDAPM Team
