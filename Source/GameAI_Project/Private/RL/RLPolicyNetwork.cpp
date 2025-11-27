// Copyright Epic Games, Inc. All Rights Reserved.

#include "RL/RLPolicyNetwork.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "Dom/JsonObject.h"
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
#include "Misc/Paths.h"
#include "Team/Objective.h"
#include "Observation/TeamObservation.h"

URLPolicyNetwork::URLPolicyNetwork()
	: bEnableExploration(true)
	, bUseONNXModel(false)
	, bCollectExperiences(true)
	, MaxExperienceBufferSize(100000)
	, bIsInitialized(false)
	, CurrentEpisodeReward(0.0f)
	, CurrentEpisodeSteps(0)
{
}

// ========================================
// Initialization
// ========================================

bool URLPolicyNetwork::Initialize(const FRLPolicyConfig& InConfig)
{
	Config = InConfig;
	bIsInitialized = true;

	UE_LOG(LogTemp, Log, TEXT("URLPolicyNetwork: Initialized with %d inputs, %d outputs"),
		Config.InputSize, Config.OutputSize);

	return true;
}

bool URLPolicyNetwork::LoadPolicy(const FString& ModelPath)
{
	Config.ModelPath = ModelPath;

	// Resolve path - support both absolute and relative paths
	FString ResolvedPath = ModelPath;
	if (!FPaths::FileExists(ResolvedPath))
	{
		// Try relative to project content directory
		ResolvedPath = FPaths::ProjectContentDir() / ModelPath;
	}
	if (!FPaths::FileExists(ResolvedPath))
	{
		// Try relative to project saved directory
		ResolvedPath = FPaths::ProjectSavedDir() / ModelPath;
	}

	if (!FPaths::FileExists(ResolvedPath))
	{
		UE_LOG(LogTemp, Error, TEXT("URLPolicyNetwork: Model file not found: %s"), *ModelPath);
		bUseONNXModel = false;
		bIsInitialized = true;
		return false;
	}

	UE_LOG(LogTemp, Log, TEXT("URLPolicyNetwork: Loading ONNX model from: %s"), *ResolvedPath);

	// Load model data from file
	TArray<uint8> ModelBytes;
	if (!FFileHelper::LoadFileToArray(ModelBytes, *ResolvedPath))
	{
		UE_LOG(LogTemp, Error, TEXT("URLPolicyNetwork: Failed to read model file"));
		bUseONNXModel = false;
		bIsInitialized = true;
		return false;
	}

	// Get NNE runtime
	TWeakInterfacePtr<INNERuntimeCPU> RuntimeCPU = UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORTCpu"));
	if (!RuntimeCPU.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("URLPolicyNetwork: NNERuntimeORTCpu not available. Using rule-based fallback."));
		bUseONNXModel = false;
		bIsInitialized = true;
		return false;
	}

	// Create model from bytes
	TObjectPtr<UNNEModelData> NewModelData = NewObject<UNNEModelData>();
	NewModelData->Init(TEXT("Onnx"), ModelBytes, TMap<FString, TConstArrayView64<uint8>>());
	TSharedPtr<UE::NNE::IModelCPU> Model = RuntimeCPU->CreateModelCPU(NewModelData);
	if (!Model.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("URLPolicyNetwork: Failed to create NNE model from ONNX data"));
		bUseONNXModel = false;
		bIsInitialized = true;
		return false;
	}

	// Create model instance
	ModelInstance = Model->CreateModelInstanceCPU();
	if (!ModelInstance.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("URLPolicyNetwork: Failed to create NNE model instance"));
		bUseONNXModel = false;
		bIsInitialized = true;
		return false;
	}

	// Get input/output tensor info
	TConstArrayView<UE::NNE::FTensorDesc> InputDescs = ModelInstance->GetInputTensorDescs();
	TConstArrayView<UE::NNE::FTensorDesc> OutputDescs = ModelInstance->GetOutputTensorDescs();

	if (InputDescs.Num() < 1 || OutputDescs.Num() < 1)
	{
		UE_LOG(LogTemp, Error, TEXT("URLPolicyNetwork: Model must have at least 1 input and 1 output"));
		ModelInstance.Reset();
		bUseONNXModel = false;
		bIsInitialized = true;
		return false;
	}

	// Log tensor info
	UE_LOG(LogTemp, Log, TEXT("URLPolicyNetwork: Model loaded successfully"));
	UE_LOG(LogTemp, Log, TEXT("  Input tensors: %d"), InputDescs.Num());
	UE_LOG(LogTemp, Log, TEXT("  Output tensors: %d"), OutputDescs.Num());

	// Setup input buffer (71 features)
	InputBuffer.SetNum(Config.InputSize);

	// Setup output buffer (16 actions + optional value head)
	// The model outputs [action_probs, state_value], but we only need action_probs
	OutputBuffer.SetNum(Config.OutputSize + 1);  // +1 for value head

	bUseONNXModel = true;
	bIsInitialized = true;

	UE_LOG(LogTemp, Log, TEXT("URLPolicyNetwork: ONNX model ready for inference"));
	return true;
}

void URLPolicyNetwork::UnloadPolicy()
{
	bIsInitialized = false;
	bUseONNXModel = false;
	Config.ModelPath = TEXT("");

	// Reset NNE model instance
	if (ModelInstance.IsValid())
	{
		ModelInstance.Reset();
	}
	ModelData = nullptr;
	InputBuffer.Empty();
	OutputBuffer.Empty();

	UE_LOG(LogTemp, Log, TEXT("URLPolicyNetwork: Policy unloaded"));
}

// ========================================
// Experience Collection (v3.0)
// ========================================

void URLPolicyNetwork::StoreExperience(const FObservationElement& State, const FTacticalAction& Action, float Reward, const FObservationElement& NextState, bool bTerminal, UObjective* CurrentObjective)
{
	if (!bCollectExperiences)
	{
		return;
	}

	// Create experience
	FRLExperience Experience(State, Action, Reward, NextState, bTerminal);
	Experience.Timestamp = FPlatformTime::Seconds();

	// Add objective embeddings
	Experience.ObjectiveEmbedding = GetObjectiveEmbedding(CurrentObjective);
	Experience.NextObjectiveEmbedding = GetObjectiveEmbedding(CurrentObjective);  // Assume same objective for now

	// Add to buffer
	CollectedExperiences.Add(Experience);

	// Update statistics
	TrainingStats.TotalExperiences++;
	CurrentEpisodeReward += Reward;
	CurrentEpisodeSteps++;

	// Handle episode termination
	if (bTerminal)
	{
		TrainingStats.EpisodesCompleted++;
		TrainingStats.LastEpisodeReward = CurrentEpisodeReward;

		if (CurrentEpisodeReward > TrainingStats.BestEpisodeReward)
		{
			TrainingStats.BestEpisodeReward = CurrentEpisodeReward;
		}

		// Update average reward (exponential moving average)
		float Alpha = 0.1f;  // Smoothing factor
		TrainingStats.AverageReward = Alpha * CurrentEpisodeReward + (1.0f - Alpha) * TrainingStats.AverageReward;

		// Update average episode length
		TrainingStats.AverageEpisodeLength = Alpha * CurrentEpisodeSteps + (1.0f - Alpha) * TrainingStats.AverageEpisodeLength;

		// Reset episode counters
		CurrentEpisodeReward = 0.0f;
		CurrentEpisodeSteps = 0;

		UE_LOG(LogTemp, Log, TEXT("URLPolicyNetwork: Episode %d complete. Reward: %.2f, Avg: %.2f, Best: %.2f"),
			TrainingStats.EpisodesCompleted,
			TrainingStats.LastEpisodeReward,
			TrainingStats.AverageReward,
			TrainingStats.BestEpisodeReward);
	}

	// Check buffer overflow
	if (CollectedExperiences.Num() > MaxExperienceBufferSize)
	{
		UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Experience buffer full (%d), removing oldest experiences"),
			MaxExperienceBufferSize);

		// Remove oldest 10% of experiences
		int32 RemoveCount = MaxExperienceBufferSize / 10;
		CollectedExperiences.RemoveAt(0, RemoveCount);
	}
}

void URLPolicyNetwork::StoreExperienceWithUncertainty(
	const FObservationElement& State,
	const FTacticalAction& Action,
	float Reward,
	const FObservationElement& NextState,
	bool bTerminal,
	UObjective* CurrentObjective,
	float MCTSValueVariance,
	float MCTSPolicyEntropy,
	float MCTSVisitCount)
{
	if (!bCollectExperiences)
	{
		return;
	}

	// Create experience
	FRLExperience Experience(State, Action, Reward, NextState, bTerminal);
	Experience.Timestamp = FPlatformTime::Seconds();

	// Add objective embeddings
	Experience.ObjectiveEmbedding = GetObjectiveEmbedding(CurrentObjective);
	Experience.NextObjectiveEmbedding = GetObjectiveEmbedding(CurrentObjective);

	// Add MCTS uncertainty metrics (Sprint 3)
	Experience.MCTSValueVariance = MCTSValueVariance;
	Experience.MCTSPolicyEntropy = MCTSPolicyEntropy;
	Experience.MCTSVisitCount = MCTSVisitCount;

	// Add to buffer
	CollectedExperiences.Add(Experience);

	// Update statistics
	TrainingStats.TotalExperiences++;
	CurrentEpisodeReward += Reward;
	CurrentEpisodeSteps++;

	// Handle episode termination
	if (bTerminal)
	{
		TrainingStats.EpisodesCompleted++;
		TrainingStats.LastEpisodeReward = CurrentEpisodeReward;

		if (CurrentEpisodeReward > TrainingStats.BestEpisodeReward)
		{
			TrainingStats.BestEpisodeReward = CurrentEpisodeReward;
		}

		// Update averages
		float Alpha = 0.1f;
		TrainingStats.AverageReward = Alpha * CurrentEpisodeReward + (1.0f - Alpha) * TrainingStats.AverageReward;
		TrainingStats.AverageEpisodeLength = Alpha * CurrentEpisodeSteps + (1.0f - Alpha) * TrainingStats.AverageEpisodeLength;

		// Reset episode counters
		CurrentEpisodeReward = 0.0f;
		CurrentEpisodeSteps = 0;

		UE_LOG(LogTemp, Log, TEXT("URLPolicyNetwork: Episode complete (with MCTS tagging). Reward: %.2f, MCTS Variance: %.3f"),
			TrainingStats.LastEpisodeReward, MCTSValueVariance);
	}

	// Check buffer overflow
	if (CollectedExperiences.Num() > MaxExperienceBufferSize)
	{
		int32 RemoveCount = MaxExperienceBufferSize / 10;
		CollectedExperiences.RemoveAt(0, RemoveCount);
	}
}

bool URLPolicyNetwork::ExportExperiencesToJSON(const FString& FilePath)
{
	if (CollectedExperiences.Num() == 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: No experiences to export"));
		return false;
	}

	// Create JSON array
	TArray<TSharedPtr<FJsonValue>> ExperiencesArray;

	for (const FRLExperience& Exp : CollectedExperiences)
	{
		TSharedPtr<FJsonObject> ExpObject = MakeShareable(new FJsonObject());

		// State (71 features)
		TArray<TSharedPtr<FJsonValue>> StateArray;
		TArray<float> StateFeatures = Exp.State.ToFeatureVector();
		for (float Feature : StateFeatures)
		{
			StateArray.Add(MakeShareable(new FJsonValueNumber(Feature)));
		}
		ExpObject->SetArrayField(TEXT("state"), StateArray);

		// Objective embedding (7 features)
		TArray<TSharedPtr<FJsonValue>> ObjectiveArray;
		for (float Feature : Exp.ObjectiveEmbedding)
		{
			ObjectiveArray.Add(MakeShareable(new FJsonValueNumber(Feature)));
		}
		ExpObject->SetArrayField(TEXT("objective_embedding"), ObjectiveArray);

		// Action (8-dimensional atomic action)
		TArray<TSharedPtr<FJsonValue>> ActionArray;
		ActionArray.Add(MakeShareable(new FJsonValueNumber(Exp.Action.MoveDirection.X)));
		ActionArray.Add(MakeShareable(new FJsonValueNumber(Exp.Action.MoveDirection.Y)));
		ActionArray.Add(MakeShareable(new FJsonValueNumber(Exp.Action.MoveSpeed)));
		ActionArray.Add(MakeShareable(new FJsonValueNumber(Exp.Action.LookDirection.X)));
		ActionArray.Add(MakeShareable(new FJsonValueNumber(Exp.Action.LookDirection.Y)));
		ActionArray.Add(MakeShareable(new FJsonValueNumber(Exp.Action.bFire ? 1.0f : 0.0f)));
		ActionArray.Add(MakeShareable(new FJsonValueNumber(Exp.Action.bCrouch ? 1.0f : 0.0f)));
		ActionArray.Add(MakeShareable(new FJsonValueNumber(Exp.Action.bUseAbility ? 1.0f : 0.0f)));
		ExpObject->SetArrayField(TEXT("action"), ActionArray);

		// Reward
		ExpObject->SetNumberField(TEXT("reward"), Exp.Reward);

		// Next state (71 features)
		TArray<TSharedPtr<FJsonValue>> NextStateArray;
		TArray<float> NextStateFeatures = Exp.NextState.ToFeatureVector();
		for (float Feature : NextStateFeatures)
		{
			NextStateArray.Add(MakeShareable(new FJsonValueNumber(Feature)));
		}
		ExpObject->SetArrayField(TEXT("next_state"), NextStateArray);

		// Next objective embedding (7 features)
		TArray<TSharedPtr<FJsonValue>> NextObjectiveArray;
		for (float Feature : Exp.NextObjectiveEmbedding)
		{
			NextObjectiveArray.Add(MakeShareable(new FJsonValueNumber(Feature)));
		}
		ExpObject->SetArrayField(TEXT("next_objective_embedding"), NextObjectiveArray);

		// Terminal
		ExpObject->SetBoolField(TEXT("terminal"), Exp.bTerminal);

		// Timestamp
		ExpObject->SetNumberField(TEXT("timestamp"), Exp.Timestamp);

		// MCTS uncertainty metrics (Sprint 3 - Curriculum Learning)
		ExpObject->SetNumberField(TEXT("mcts_value_variance"), Exp.MCTSValueVariance);
		ExpObject->SetNumberField(TEXT("mcts_policy_entropy"), Exp.MCTSPolicyEntropy);
		ExpObject->SetNumberField(TEXT("mcts_visit_count"), Exp.MCTSVisitCount);

		ExperiencesArray.Add(MakeShareable(new FJsonValueObject(ExpObject)));
	}

	// Create root object with metadata
	TSharedPtr<FJsonObject> RootObject = MakeShareable(new FJsonObject());
	RootObject->SetArrayField(TEXT("experiences"), ExperiencesArray);
	RootObject->SetNumberField(TEXT("total_experiences"), CollectedExperiences.Num());
	RootObject->SetNumberField(TEXT("episodes_completed"), TrainingStats.EpisodesCompleted);
	RootObject->SetNumberField(TEXT("average_reward"), TrainingStats.AverageReward);
	RootObject->SetNumberField(TEXT("best_reward"), TrainingStats.BestEpisodeReward);

	// Serialize to JSON string
	FString OutputString;
	TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutputString);
	if (!FJsonSerializer::Serialize(RootObject.ToSharedRef(), Writer))
	{
		UE_LOG(LogTemp, Error, TEXT("URLPolicyNetwork: Failed to serialize experiences to JSON"));
		return false;
	}

	// Write to file
	FString SafeDirectory = FPaths::ProjectSavedDir() / TEXT("Experiences");
	FString SafeFilePath = SafeDirectory / TEXT("experiences.json");

	UE_LOG(LogTemp, Log, TEXT("URLPolicyNetwork: Using safe path: %s"), *SafeFilePath);

	IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

	if (!PlatformFile.DirectoryExists(*SafeDirectory))
	{
		if (!PlatformFile.CreateDirectoryTree(*SafeDirectory))
		{
			UE_LOG(LogTemp, Error, TEXT("URLPolicyNetwork: Failed to create directory: %s"), *SafeDirectory);
			return false;
		}
	}

	// 파일 저장
	if (!FFileHelper::SaveStringToFile(OutputString, *SafeFilePath))
	{
		UE_LOG(LogTemp, Error, TEXT("URLPolicyNetwork: Failed to write file"));
		return false;
	}

	return true;
}

void URLPolicyNetwork::ClearExperiences()
{
	CollectedExperiences.Empty();
	UE_LOG(LogTemp, Log, TEXT("URLPolicyNetwork: Cleared all experiences"));
}

// ========================================
// Statistics
// ========================================

void URLPolicyNetwork::ResetStatistics()
{
	TrainingStats = FRLTrainingStats();
	CurrentEpisodeReward = 0.0f;
	CurrentEpisodeSteps = 0;

	UE_LOG(LogTemp, Log, TEXT("URLPolicyNetwork: Reset statistics"));
}

void URLPolicyNetwork::UpdateEpsilon()
{
	Config.Epsilon = FMath::Max(Config.Epsilon * Config.EpsilonDecay, Config.MinEpsilon);
}

// ========================================
// Neural Network Inference
// ========================================

TArray<float> URLPolicyNetwork::ForwardPass(const TArray<float>& InputFeatures)
{
	if (!ModelInstance.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Model instance not valid, returning zero action"));
		TArray<float> ZeroAction;
		ZeroAction.Init(0.0f, Config.OutputSize);
		return ZeroAction;
	}

	// Copy input features to buffer
	if (InputFeatures.Num() != Config.InputSize)
	{
		UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Input size mismatch (got %d, expected %d)"),
			InputFeatures.Num(), Config.InputSize);
		TArray<float> ZeroAction;
		ZeroAction.Init(0.0f, Config.OutputSize);
		return ZeroAction;
	}

	// Prepare input tensor binding
	InputBuffer = InputFeatures;

	// Create tensor shapes for binding
	UE::NNE::FTensorShape InputTensorShape = UE::NNE::FTensorShape::Make({ 1u, static_cast<uint32>(Config.InputSize) });

	// Create input tensor bindings
	TArray<UE::NNE::FTensorBindingCPU> InputBindings;
	InputBindings.Add(UE::NNE::FTensorBindingCPU{
		InputBuffer.GetData(),
		static_cast<uint64>(InputBuffer.Num() * sizeof(float))
	});

	// Create output tensor bindings
	// First output: action_probabilities (16 values)
	// Second output: state_value (1 value)
	TArray<float> ActionProbsBuffer;
	TArray<float> StateValueBuffer;
	ActionProbsBuffer.SetNum(Config.OutputSize);
	StateValueBuffer.SetNum(1);

	TArray<UE::NNE::FTensorBindingCPU> OutputBindings;
	OutputBindings.Add(UE::NNE::FTensorBindingCPU{
		ActionProbsBuffer.GetData(),
		static_cast<uint64>(ActionProbsBuffer.Num() * sizeof(float))
	});
	OutputBindings.Add(UE::NNE::FTensorBindingCPU{
		StateValueBuffer.GetData(),
		static_cast<uint64>(StateValueBuffer.Num() * sizeof(float))
	});

	// Set input tensor shapes
	TArray<UE::NNE::FTensorShape> InputShapes;
	InputShapes.Add(InputTensorShape);

	UE::NNE::EResultStatus SetInputStatus = ModelInstance->SetInputTensorShapes(InputShapes);
	if (SetInputStatus != UE::NNE::EResultStatus::Ok)
	{
		UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Failed to set input tensor shapes"));
		TArray<float> ZeroAction;
		ZeroAction.Init(0.0f, Config.OutputSize);
		return ZeroAction;
	}

	// Run inference
	UE::NNE::EResultStatus RunStatus = ModelInstance->RunSync(InputBindings, OutputBindings);

	if (RunStatus != UE::NNE::EResultStatus::Ok)
	{
		UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Inference failed, returning zero action"));
		TArray<float> ZeroAction;
		ZeroAction.Init(0.0f, Config.OutputSize);
		return ZeroAction;
	}

	// Model already outputs softmax probabilities, return directly
	return ActionProbsBuffer;
}

TArray<float> URLPolicyNetwork::Softmax(const TArray<float>& Logits)
{
	TArray<float> Probabilities;
	Probabilities.SetNum(Logits.Num());

	// Find max logit for numerical stability
	float MaxLogit = -MAX_FLT;
	for (float Logit : Logits)
	{
		if (Logit > MaxLogit)
		{
			MaxLogit = Logit;
		}
	}

	// Compute exp(logit - max) and sum
	float SumExp = 0.0f;
	for (int32 i = 0; i < Logits.Num(); i++)
	{
		Probabilities[i] = FMath::Exp(Logits[i] - MaxLogit);
		SumExp += Probabilities[i];
	}

	// Normalize
	for (int32 i = 0; i < Probabilities.Num(); i++)
	{
		Probabilities[i] /= SumExp;
	}

	return Probabilities;
}

// ========================================
// Atomic Action Space (v3.0)
// ========================================

FTacticalAction URLPolicyNetwork::GetAction(const FObservationElement& Observation, UObjective* CurrentObjective)
{
	FActionSpaceMask DefaultMask;  // No constraints
	return GetActionWithMask(Observation, CurrentObjective, DefaultMask);
}

FTacticalAction URLPolicyNetwork::GetActionWithMask(const FObservationElement& Observation, UObjective* CurrentObjective, const FActionSpaceMask& Mask)
{
	if (!bIsInitialized)
	{
		UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Not initialized, returning default action"));
		FTacticalAction DefaultAction;
		return DefaultAction;
	}

	if (bUseONNXModel && ModelInstance.IsValid())
	{
		// Build enhanced input: 71 observation + 7 objective embedding = 78 features
		TArray<float> InputFeatures = Observation.ToFeatureVector();
		TArray<float> ObjectiveEmbed = GetObjectiveEmbedding(CurrentObjective);
		InputFeatures.Append(ObjectiveEmbed);

		// Forward pass (expects 8-dim output: move_x, move_y, speed, look_x, look_y, fire, crouch, ability)
		TArray<float> NetworkOutput = ForwardPass(InputFeatures);

		// Convert to action
		FTacticalAction Action = NetworkOutputToAction(NetworkOutput);

		// Apply spatial constraints
		return ApplyMask(Action, Mask);
	}
	else
	{
		// Rule-based fallback
		FTacticalAction Action = GetActionRuleBased(Observation, CurrentObjective);
		return ApplyMask(Action, Mask);
	}
}

float URLPolicyNetwork::GetStateValue(const FObservationElement& Observation, UObjective* CurrentObjective)
{
	// TODO: Load and use PPO critic network (team_value_network.onnx) for value estimation
	// For now, use a simple heuristic based on observation features

	if (!bIsInitialized)
	{
		return 0.0f;
	}

	// Build enhanced input: 71 observation + 7 objective embedding = 78 features
	TArray<float> InputFeatures = Observation.ToFeatureVector();
	TArray<float> ObjectiveEmbed = GetObjectiveEmbedding(CurrentObjective);
	InputFeatures.Append(ObjectiveEmbed);

	// If ONNX model is loaded, try to use critic network
	// NOTE: Current model only has actor head, critic head will be added in next training
	if (bUseONNXModel && ModelInstance.IsValid())
	{
		// Future: Export critic head separately and load here
		// For now, use heuristic value estimation
		UE_LOG(LogTemp, Verbose, TEXT("URLPolicyNetwork: Critic network not yet loaded, using heuristic value"));
	}

	// Heuristic value estimation (temporary until critic network is loaded)
	float Value = 0.0f;

	// Health component: +1.0 at full health, -1.0 at zero health
	Value += (Observation.AgentHealth - 50.0f) / 50.0f;  // Normalized to [-1, 1]

	// Enemy threat penalty
	Value -= Observation.VisibleEnemyCount * 0.2f;

	// Cover bonus
	if (Observation.bHasCover)
	{
		Value += 0.3f;
	}

	// Ammo consideration
	if (Observation.CurrentAmmo < 10.0f)
	{
		Value -= 0.2f;
	}

	return FMath::Clamp(Value, -1.0f, 1.0f);
}

TArray<float> URLPolicyNetwork::GetObjectivePriors(const FTeamObservation& TeamObs)
{
	// v3.0 Sprint 4: Heuristic-based priors to guide MCTS
	// These priors are calculated based on team state to provide intelligent initial guidance
	// Future: Replace with learned priors from dual-head policy network

	TArray<float> Priors;
	Priors.Init(0.1f, 7);  // Start with small baseline probability

	// Objective type indices (matching EObjectiveType enum)
	const int32 ELIMINATE = 0;
	const int32 CAPTURE_OBJ = 1;
	const int32 DEFEND_OBJ = 2;
	const int32 SUPPORT_ALLY = 3;
	const int32 FORMATION_MOVE = 4;
	const int32 RETREAT = 5;
	const int32 RESCUE_ALLY = 6;

	// Context-aware prior assignment
	if (TeamObs.TotalVisibleEnemies > 0)
	{
		// COMBAT SITUATION
		if (TeamObs.bOutnumbered && TeamObs.AverageTeamHealth < 50.0f)
		{
			// Outnumbered and weak → retreat highly preferred
			Priors[RETREAT] = 0.4f;
			Priors[DEFEND_OBJ] = 0.2f;
			Priors[SUPPORT_ALLY] = 0.15f;
			Priors[ELIMINATE] = 0.05f;
		}
		else if (TeamObs.bFlanked)
		{
			// Being flanked → defensive posture + support
			Priors[DEFEND_OBJ] = 0.3f;
			Priors[SUPPORT_ALLY] = 0.25f;
			Priors[FORMATION_MOVE] = 0.2f;  // Regroup
			Priors[ELIMINATE] = 0.1f;
		}
		else if (TeamObs.AverageTeamHealth > 70.0f && !TeamObs.bOutnumbered)
		{
			// Strong position → aggressive
			Priors[ELIMINATE] = 0.35f;
			Priors[CAPTURE_OBJ] = 0.25f;
			Priors[SUPPORT_ALLY] = 0.15f;
			Priors[DEFEND_OBJ] = 0.1f;
		}
		else
		{
			// Balanced combat → mixed tactics
			Priors[ELIMINATE] = 0.25f;
			Priors[DEFEND_OBJ] = 0.2f;
			Priors[SUPPORT_ALLY] = 0.2f;
			Priors[CAPTURE_OBJ] = 0.15f;
		}
	}
	else
	{
		// NO ENEMIES VISIBLE
		if (TeamObs.AverageTeamHealth < 40.0f)
		{
			// Low health, no enemies → recover and defend
			Priors[DEFEND_OBJ] = 0.35f;
			Priors[RESCUE_ALLY] = 0.25f;
			Priors[FORMATION_MOVE] = 0.2f;
		}
		else if (TeamObs.DistanceToObjective > 1000.0f)
		{
			// Far from objective → advance and capture
			Priors[FORMATION_MOVE] = 0.35f;
			Priors[CAPTURE_OBJ] = 0.3f;
			Priors[DEFEND_OBJ] = 0.15f;
		}
		else if (TeamObs.FormationCoherence < 0.5f)
		{
			// Formation broken → regroup
			Priors[FORMATION_MOVE] = 0.4f;
			Priors[DEFEND_OBJ] = 0.2f;
			Priors[CAPTURE_OBJ] = 0.2f;
		}
		else
		{
			// Stable situation → objective-focused
			Priors[CAPTURE_OBJ] = 0.35f;
			Priors[DEFEND_OBJ] = 0.25f;
			Priors[FORMATION_MOVE] = 0.2f;
		}
	}

	// Normalize to sum to 1.0
	float Sum = 0.0f;
	for (float Prior : Priors)
	{
		Sum += Prior;
	}
	if (Sum > 0.0f)
	{
		for (int32 i = 0; i < Priors.Num(); ++i)
		{
			Priors[i] /= Sum;
		}
	}

	UE_LOG(LogTemp, Verbose, TEXT("RL Policy Priors: Eliminate=%.2f, Capture=%.2f, Defend=%.2f, Support=%.2f, Move=%.2f, Retreat=%.2f, Rescue=%.2f"),
		Priors[ELIMINATE], Priors[CAPTURE_OBJ], Priors[DEFEND_OBJ], Priors[SUPPORT_ALLY],
		Priors[FORMATION_MOVE], Priors[RETREAT], Priors[RESCUE_ALLY]);

	return Priors;
}

FTacticalAction URLPolicyNetwork::NetworkOutputToAction(const TArray<float>& NetworkOutput)
{
	FTacticalAction Action;

	// Network output format: [move_x, move_y, speed, look_x, look_y, fire_logit, crouch_logit, ability_logit]
	if (NetworkOutput.Num() >= 8)
	{
		// Continuous actions (already normalized to [-1,1] or [0,1] by network)
		Action.MoveDirection.X = FMath::Clamp(NetworkOutput[0], -1.0f, 1.0f);
		Action.MoveDirection.Y = FMath::Clamp(NetworkOutput[1], -1.0f, 1.0f);
		Action.MoveSpeed = FMath::Clamp(NetworkOutput[2], 0.0f, 1.0f);
		Action.LookDirection.X = FMath::Clamp(NetworkOutput[3], -1.0f, 1.0f);
		Action.LookDirection.Y = FMath::Clamp(NetworkOutput[4], -1.0f, 1.0f);

		// Discrete actions (use sigmoid threshold: > 0.5 = true)
		Action.bFire = NetworkOutput[5] > 0.5f;
		Action.bCrouch = NetworkOutput[6] > 0.5f;
		Action.bUseAbility = NetworkOutput[7] > 0.5f;

		// AbilityID (if more outputs exist)
		if (NetworkOutput.Num() > 8)
		{
			Action.AbilityID = FMath::RoundToInt(NetworkOutput[8]);
		}
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Invalid network output size %d, expected 8+"), NetworkOutput.Num());
	}

	return Action;
}

FTacticalAction URLPolicyNetwork::ApplyMask(const FTacticalAction& Action, const FActionSpaceMask& Mask)
{
	FTacticalAction ConstrainedAction = Action;

	// Movement constraints
	if (Mask.bLockMovementX)
	{
		ConstrainedAction.MoveDirection.X = 0.0f;
	}
	if (Mask.bLockMovementY)
	{
		ConstrainedAction.MoveDirection.Y = 0.0f;
	}

	// Speed constraints
	ConstrainedAction.MoveSpeed = FMath::Min(ConstrainedAction.MoveSpeed, Mask.MaxSpeed);

	// Sprint constraints
	if (!Mask.bCanSprint && ConstrainedAction.MoveSpeed > 0.6f)
	{
		ConstrainedAction.MoveSpeed = 0.6f;  // Limit to walk speed
	}

	// Aiming constraints (convert 2D direction to angles)
	// Note: Full angle constraint implementation would require current agent rotation
	// For now, we just clamp the look direction vector
	ConstrainedAction.LookDirection.X = FMath::Clamp(ConstrainedAction.LookDirection.X, -1.0f, 1.0f);
	ConstrainedAction.LookDirection.Y = FMath::Clamp(ConstrainedAction.LookDirection.Y, -1.0f, 1.0f);

	// Crouch constraints
	if (Mask.bForceCrouch)
	{
		ConstrainedAction.bCrouch = true;
	}

	// Safety lock (disable firing)
	if (Mask.bSafetyLock)
	{
		ConstrainedAction.bFire = false;
	}

	return ConstrainedAction;
}

FTacticalAction URLPolicyNetwork::GetActionRuleBased(const FObservationElement& Observation, UObjective* CurrentObjective)
{
	FTacticalAction Action;

	// Extract key features
	float Health = Observation.AgentHealth;
	int32 VisibleEnemies = Observation.VisibleEnemyCount;
	bool bHasCover = Observation.bHasCover;
	float NearestCoverDistance = Observation.NearestCoverDistance;

	// Calculate nearest enemy direction
	FVector2D EnemyDirection = FVector2D::ZeroVector;
	float NearestEnemyDistance = MAX_FLT;
	if (Observation.NearbyEnemies.Num() > 0)
	{
		NearestEnemyDistance = Observation.NearbyEnemies[0].Distance;
		// Approximate direction from bearing (simplified)
		float Bearing = Observation.NearbyEnemies[0].RelativeAngle;
		EnemyDirection.X = FMath::Sin(FMath::DegreesToRadians(Bearing));
		EnemyDirection.Y = FMath::Cos(FMath::DegreesToRadians(Bearing));
	}

	// Rule-based movement
	if (Health < 30.0f)
	{
		// Low health: retreat away from enemies
		Action.MoveDirection = -EnemyDirection;  // Move away
		Action.MoveSpeed = 1.0f;  // Sprint
		Action.bCrouch = false;
	}
	else if (!bHasCover && VisibleEnemies > 0 && NearestCoverDistance < 500.0f)
	{
		// No cover, seek it (move perpendicular to enemy)
		Action.MoveDirection = FVector2D(-EnemyDirection.Y, EnemyDirection.X);  // Perpendicular
		Action.MoveSpeed = 0.8f;
		Action.bCrouch = false;
	}
	else if (VisibleEnemies > 0)
	{
		// Enemies visible: cautious advance or hold
		if (Health > 70.0f)
		{
			Action.MoveDirection = EnemyDirection * 0.3f;  // Slow advance
			Action.MoveSpeed = 0.4f;
		}
		else
		{
			Action.MoveDirection = FVector2D::ZeroVector;  // Hold position
			Action.MoveSpeed = 0.0f;
		}
		Action.bCrouch = true;  // Use cover
	}
	else
	{
		// No enemies: patrol forward
		Action.MoveDirection = FVector2D(0.0f, 1.0f);  // Forward
		Action.MoveSpeed = 0.5f;
		Action.bCrouch = false;
	}

	// Rule-based aiming
	if (VisibleEnemies > 0)
	{
		// Aim at nearest enemy
		Action.LookDirection = EnemyDirection;
		Action.bFire = (NearestEnemyDistance < 1000.0f);  // Fire if in range
	}
	else
	{
		// Look forward
		Action.LookDirection = FVector2D(0.0f, 1.0f);
		Action.bFire = false;
	}

	// Ability usage (simple heuristic)
	Action.bUseAbility = (Health < 50.0f && VisibleEnemies > 2);  // Use ability when outnumbered
	Action.AbilityID = 0;

	UE_LOG(LogTemp, Warning, TEXT("🔧 [RULE-BASED] Action: Move=(%.2f,%.2f) Speed=%.2f Look=(%.2f,%.2f) Fire=%d Crouch=%d"),
		Action.MoveDirection.X, Action.MoveDirection.Y, Action.MoveSpeed,
		Action.LookDirection.X, Action.LookDirection.Y, Action.bFire, Action.bCrouch);

	return Action;
}

TArray<float> URLPolicyNetwork::GetObjectiveEmbedding(UObjective* CurrentObjective)
{
	// 7-element one-hot encoding for objective type
	TArray<float> Embedding;
	Embedding.Init(0.0f, 7);

	if (CurrentObjective)
	{
		// Get objective type and encode as one-hot
		EObjectiveType ObjType = CurrentObjective->Type;

		switch (ObjType)
		{
			case EObjectiveType::Eliminate:
				Embedding[0] = 1.0f;
				break;
			case EObjectiveType::CaptureObjective:
				Embedding[1] = 1.0f;
				break;
			case EObjectiveType::DefendObjective:
				Embedding[2] = 1.0f;
				break;
			case EObjectiveType::SupportAlly:
				Embedding[3] = 1.0f;
				break;
			case EObjectiveType::FormationMove:
				Embedding[4] = 1.0f;
				break;
			case EObjectiveType::Retreat:
				Embedding[5] = 1.0f;
				break;
			case EObjectiveType::RescueAlly:
				Embedding[6] = 1.0f;
				break;
			default:
				// None or unknown - leave as zeros
				break;
		}
	}
	// If null objective, return all zeros (no objective)

	return Embedding;
}
