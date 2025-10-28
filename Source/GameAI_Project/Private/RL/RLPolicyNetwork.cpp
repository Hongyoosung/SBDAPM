// Copyright Epic Games, Inc. All Rights Reserved.

#include "RL/RLPolicyNetwork.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "Dom/JsonObject.h"

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

	// TODO: Implement ONNX model loading
	// For now, log a warning and fall back to rule-based
	UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: ONNX loading not yet implemented. Using rule-based fallback."));
	UE_LOG(LogTemp, Warning, TEXT("  Model path: %s"), *ModelPath);

	bUseONNXModel = false;
	bIsInitialized = true;

	return false;  // Return false until ONNX is implemented
}

void URLPolicyNetwork::UnloadPolicy()
{
	bIsInitialized = false;
	bUseONNXModel = false;
	Config.ModelPath = TEXT("");

	UE_LOG(LogTemp, Log, TEXT("URLPolicyNetwork: Policy unloaded"));
}

// ========================================
// Inference
// ========================================

ETacticalAction URLPolicyNetwork::SelectAction(const FObservationElement& Observation)
{
	if (!bIsInitialized)
	{
		UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: Not initialized, returning default action"));
		return ETacticalAction::DefensiveHold;
	}

	// Epsilon-greedy exploration
	if (bEnableExploration && FMath::FRand() < Config.Epsilon)
	{
		ETacticalAction RandomAction = GetRandomAction();
		UE_LOG(LogTemp, Verbose, TEXT("URLPolicyNetwork: Exploring - selected random action: %s"),
			*GetActionName(RandomAction));
		return RandomAction;
	}

	// Get action probabilities
	TArray<float> Probabilities = GetActionProbabilities(Observation);

	// Select greedy action
	ETacticalAction SelectedAction = GetGreedyAction(Probabilities);

	UE_LOG(LogTemp, Verbose, TEXT("URLPolicyNetwork: Exploiting - selected action: %s (prob: %.3f)"),
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
	// TODO: Implement ONNX forward pass
	// For now, return rule-based probabilities

	UE_LOG(LogTemp, Warning, TEXT("URLPolicyNetwork: ONNX forward pass not implemented, using rule-based fallback"));

	// Create dummy observation from features
	FObservationElement DummyObs;
	// ... (would need to reconstruct observation from features, skip for now)

	return GetRuleBasedProbabilities(DummyObs);
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

	// Rule 4: Enemies very close → Hold position or retreat
	if (NearestEnemyDistance < 200.0f)
	{
		Probabilities[ActionToIndex(ETacticalAction::DefensiveHold)] += 5.0f;
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
