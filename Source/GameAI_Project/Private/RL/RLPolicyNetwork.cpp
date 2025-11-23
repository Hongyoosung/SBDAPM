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
// Inference
// ========================================

ETacticalAction URLPolicyNetwork::SelectAction(const FObservationElement& Observation)
{
	UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Selecting action for observation"));
	if (!bIsInitialized)
	{
		UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Not initialized, returning default action"));
		return ETacticalAction::DefensiveHold;
	}

	// Epsilon-greedy exploration
	if (bEnableExploration && FMath::FRand() < Config.Epsilon)
	{
		ETacticalAction RandomAction = GetRandomAction();
		UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Exploring - selected random action: %s"),
			*GetActionName(RandomAction));
		return RandomAction;
	}

	// Get action probabilities
	TArray<float> Probabilities = GetActionProbabilities(Observation);

	// Select greedy action
	ETacticalAction SelectedAction = GetGreedyAction(Probabilities);

	UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Exploiting - selected action: %s (prob: %.3f)"),
		*GetActionName(SelectedAction),
		Probabilities[ActionToIndex(SelectedAction)]);

	return SelectedAction;
}

TArray<float> URLPolicyNetwork::GetActionProbabilities(const FObservationElement& Observation)
{
	if (!bIsInitialized)
	{
		// Return uniform distribution if not initialized
		TArray<float> UniformProbs;
		UniformProbs.SetNum(Config.OutputSize);
		float UniformValue = 1.0f / Config.OutputSize;
		for (int32 i = 0; i < Config.OutputSize; i++)
		{
			UniformProbs[i] = UniformValue;
		}
		return UniformProbs;
	}

	if (bUseONNXModel)
	{
		// ONNX inference
		TArray<float> InputFeatures = Observation.ToFeatureVector();
		return ForwardPass(InputFeatures);
	}
	else
	{
		// Rule-based fallback
		return GetRuleBasedProbabilities(Observation);
	}
}

float URLPolicyNetwork::GetActionValue(const FObservationElement& Observation, ETacticalAction Action)
{
	TArray<float> Probabilities = GetActionProbabilities(Observation);
	int32 ActionIndex = ActionToIndex(Action);

	if (ActionIndex >= 0 && ActionIndex < Probabilities.Num())
	{
		return Probabilities[ActionIndex];
	}

	return 0.0f;
}

// ========================================
// Experience Collection
// ========================================

void URLPolicyNetwork::StoreExperience(const FObservationElement& State, ETacticalAction Action, float Reward, const FObservationElement& NextState, bool bTerminal)
{
	if (!bCollectExperiences)
	{
		return;
	}

	// Create experience
	FRLExperience Experience(State, Action, Reward, NextState, bTerminal);
	Experience.Timestamp = FPlatformTime::Seconds();

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

		// Action (integer index)
		ExpObject->SetNumberField(TEXT("action"), ActionToIndex(Exp.Action));

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

		// Terminal
		ExpObject->SetBoolField(TEXT("terminal"), Exp.bTerminal);

		// Timestamp
		ExpObject->SetNumberField(TEXT("timestamp"), Exp.Timestamp);

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

FString URLPolicyNetwork::GetActionName(ETacticalAction Action)
{
	switch (Action)
	{
		case ETacticalAction::AggressiveAssault: return TEXT("Aggressive Assault");
		case ETacticalAction::CautiousAdvance: return TEXT("Cautious Advance");
		case ETacticalAction::DefensiveHold: return TEXT("Defensive Hold");
		case ETacticalAction::TacticalRetreat: return TEXT("Tactical Retreat");
		case ETacticalAction::SeekCover: return TEXT("Seek Cover");
		case ETacticalAction::FlankLeft: return TEXT("Flank Left");
		case ETacticalAction::FlankRight: return TEXT("Flank Right");
		case ETacticalAction::MaintainDistance: return TEXT("Maintain Distance");
		case ETacticalAction::SuppressiveFire: return TEXT("Suppressive Fire");
		case ETacticalAction::ProvideCoveringFire: return TEXT("Provide Covering Fire");
		case ETacticalAction::Reload: return TEXT("Reload");
		case ETacticalAction::UseAbility: return TEXT("Use Ability");
		case ETacticalAction::Sprint: return TEXT("Sprint");
		case ETacticalAction::Crouch: return TEXT("Crouch");
		case ETacticalAction::Patrol: return TEXT("Patrol");
		case ETacticalAction::Hold: return TEXT("Hold");
		default: return TEXT("Unknown");
	}
}

// ========================================
// Neural Network Inference
// ========================================

TArray<float> URLPolicyNetwork::ForwardPass(const TArray<float>& InputFeatures)
{
	if (!ModelInstance.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Model instance not valid, using rule-based fallback"));
		FObservationElement DummyObs;
		return GetRuleBasedProbabilities(DummyObs);
	}

	// Copy input features to buffer
	if (InputFeatures.Num() != Config.InputSize)
	{
		UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Input size mismatch (got %d, expected %d)"),
			InputFeatures.Num(), Config.InputSize);
		FObservationElement DummyObs;
		return GetRuleBasedProbabilities(DummyObs);
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
		FObservationElement DummyObs;
		return GetRuleBasedProbabilities(DummyObs);
	}

	// Run inference
	UE::NNE::EResultStatus RunStatus = ModelInstance->RunSync(InputBindings, OutputBindings);

	if (RunStatus != UE::NNE::EResultStatus::Ok)
	{
		UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Inference failed, using rule-based fallback"));
		FObservationElement DummyObs;
		return GetRuleBasedProbabilities(DummyObs);
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
// Rule-Based Fallback
// ========================================

ETacticalAction URLPolicyNetwork::SelectActionRuleBased(const FObservationElement& Observation)
{
	TArray<float> Probabilities = GetRuleBasedProbabilities(Observation);
	return GetGreedyAction(Probabilities);
}

TArray<float> URLPolicyNetwork::GetRuleBasedProbabilities(const FObservationElement& Observation)
{
	UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Using rule-based action selection"));
	// Initialize probabilities (16 actions)
	TArray<float> Probabilities;
	Probabilities.Init(0.1f, 16);  // Small baseline probability for all actions

	// Extract key features
	float Health = Observation.AgentHealth;
	int32 VisibleEnemies = Observation.VisibleEnemyCount;
	bool bHasCover = Observation.bHasCover;
	float NearestCoverDistance = Observation.NearestCoverDistance;

	// Calculate nearest enemy distance
	float NearestEnemyDistance = MAX_FLT;
	if (Observation.NearbyEnemies.Num() > 0)
	{
		NearestEnemyDistance = Observation.NearbyEnemies[0].Distance;
	}

	// Rule 1: Low health → Seek cover or retreat
	if (Health < 30.0f)
	{
		Probabilities[ActionToIndex(ETacticalAction::TacticalRetreat)] += 5.0f;
		Probabilities[ActionToIndex(ETacticalAction::SeekCover)] += 4.0f;
		Probabilities[ActionToIndex(ETacticalAction::DefensiveHold)] += 2.0f;
		UE_LOG(LogTemp, Warning, TEXT("[RL POLICY] Rule 1 triggered: Low health (%.1f%%) → Retreat/Cover/DefensiveHold"), Health);
	}

	// Rule 2: No cover and enemies visible → Seek cover
	if (!bHasCover && VisibleEnemies > 0 && NearestCoverDistance < 500.0f)
	{
		Probabilities[ActionToIndex(ETacticalAction::SeekCover)] += 6.0f;
		Probabilities[ActionToIndex(ETacticalAction::Sprint)] += 3.0f;
	}

	// Rule 3: Healthy + cover + enemies visible → Aggressive tactics
	if (Health > 70.0f && bHasCover && VisibleEnemies > 0)
	{
		Probabilities[ActionToIndex(ETacticalAction::AggressiveAssault)] += 4.0f;
		Probabilities[ActionToIndex(ETacticalAction::SuppressiveFire)] += 3.0f;
		Probabilities[ActionToIndex(ETacticalAction::FlankLeft)] += 2.0f;
		Probabilities[ActionToIndex(ETacticalAction::FlankRight)] += 2.0f;
	}

	// Rule 4: Enemies very close → Aggressive tactics (changed from defensive)
	if (NearestEnemyDistance < 200.0f)
	{
		Probabilities[ActionToIndex(ETacticalAction::AggressiveAssault)] += 5.0f;  // Changed from DefensiveHold
		UE_LOG(LogTemp, Warning, TEXT("[RL POLICY] Rule 4 triggered: Enemy very close (%.1f units) → AggressiveAssault"), NearestEnemyDistance);
		if (Health < 50.0f)
		{
			Probabilities[ActionToIndex(ETacticalAction::TacticalRetreat)] += 4.0f;
		}
	}

	// Rule 5: No enemies visible → Patrol or cautious advance
	if (VisibleEnemies == 0)
	{
		Probabilities[ActionToIndex(ETacticalAction::Patrol)] += 3.0f;
		Probabilities[ActionToIndex(ETacticalAction::CautiousAdvance)] += 2.0f;
		Probabilities[ActionToIndex(ETacticalAction::Hold)] += 2.0f;
		UE_LOG(LogTemp, Warning, TEXT("[RL POLICY] Rule 5 triggered: No enemies visible → Patrol/CautiousAdvance/Hold"));
	}

	// Rule 6: Multiple enemies → Seek cover + suppressive fire
	if (VisibleEnemies >= 3)
	{
		Probabilities[ActionToIndex(ETacticalAction::SeekCover)] += 4.0f;
		Probabilities[ActionToIndex(ETacticalAction::SuppressiveFire)] += 4.0f;
		Probabilities[ActionToIndex(ETacticalAction::ProvideCoveringFire)] += 3.0f;
	}

	// Normalize to probabilities
	return Softmax(Probabilities);
}

// ========================================
// Helper Functions
// ========================================

ETacticalAction URLPolicyNetwork::SampleAction(const TArray<float>& Probabilities)
{
	float RandomValue = FMath::FRand();
	float CumulativeProbability = 0.0f;

	for (int32 i = 0; i < Probabilities.Num(); i++)
	{
		CumulativeProbability += Probabilities[i];
		if (RandomValue <= CumulativeProbability)
		{
			return IndexToAction(i);
		}
	}

	// Fallback (should never reach here)
	return IndexToAction(Probabilities.Num() - 1);
}

ETacticalAction URLPolicyNetwork::GetGreedyAction(const TArray<float>& Probabilities)
{
	int32 BestActionIndex = 0;
	float BestProbability = -1.0f;

	for (int32 i = 0; i < Probabilities.Num(); i++)
	{
		if (Probabilities[i] > BestProbability)
		{
			BestProbability = Probabilities[i];
			BestActionIndex = i;
		}
	}

	return IndexToAction(BestActionIndex);
}

ETacticalAction URLPolicyNetwork::GetRandomAction()
{
	int32 RandomIndex = FMath::RandRange(0, Config.OutputSize - 1);
	return IndexToAction(RandomIndex);
}

int32 URLPolicyNetwork::ActionToIndex(ETacticalAction Action)
{
	return static_cast<int32>(Action);
}

ETacticalAction URLPolicyNetwork::IndexToAction(int32 Index)
{
	if (Index < 0 || Index >= 16)
	{
		UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Invalid action index %d, returning default"), Index);
		return ETacticalAction::DefensiveHold;
	}

	return static_cast<ETacticalAction>(Index);
}
