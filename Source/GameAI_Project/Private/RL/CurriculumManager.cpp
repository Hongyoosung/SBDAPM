// Copyright Epic Games, Inc. All Rights Reserved.

#include "RL/CurriculumManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

UCurriculumManager::UCurriculumManager()
{
	RandomStream.Initialize(FMath::Rand());
}

void UCurriculumManager::AddScenario(const FMCTSScenarioMetrics& Scenario)
{
	// Add to buffer
	Scenarios.Add(Scenario);

	// Update total priority
	TotalPriority += Scenario.CalculatePriority();

	// Prune if buffer full
	if (Scenarios.Num() > MaxBufferSize)
	{
		PruneOldScenarios(MaxBufferSize);
	}
}

void UCurriculumManager::UpdateScenarioOutcome(int32 ScenarioIndex, float ActualReward)
{
	if (Scenarios.IsValidIndex(ScenarioIndex))
	{
		Scenarios[ScenarioIndex].ActualReward = ActualReward;
	}
}

TArray<FMCTSScenarioMetrics> UCurriculumManager::SampleScenarios(int32 BatchSize, bool bUsePrioritization)
{
	TArray<FMCTSScenarioMetrics> Sampled;

	if (Scenarios.Num() == 0)
	{
		return Sampled;
	}

	BatchSize = FMath::Min(BatchSize, Scenarios.Num());
	Sampled.Reserve(BatchSize);

	if (!bUsePrioritization)
	{
		// Uniform random sampling
		TArray<int32> Indices;
		for (int32 i = 0; i < Scenarios.Num(); ++i)
		{
			Indices.Add(i);
		}

		// Shuffle and take first BatchSize
		for (int32 i = 0; i < BatchSize; ++i)
		{
			int32 SwapIdx = RandomStream.RandRange(i, Indices.Num() - 1);
			Indices.Swap(i, SwapIdx);
			Sampled.Add(Scenarios[Indices[i]]);
		}
	}
	else
	{
		// Prioritized sampling
		RebuildPriorityDistribution();

		// Sample with replacement based on priority
		for (int32 i = 0; i < BatchSize; ++i)
		{
			float RandomValue = RandomStream.FRand() * TotalPriority;
			float CumulativePriority = 0.0f;

			for (int32 j = 0; j < Scenarios.Num(); ++j)
			{
				float Priority = FMath::Pow(Scenarios[j].CalculatePriority(), PrioritizationExponent);
				CumulativePriority += Priority;

				if (CumulativePriority >= RandomValue)
				{
					Sampled.Add(Scenarios[j]);
					break;
				}
			}
		}
	}

	return Sampled;
}

void UCurriculumManager::GetStatistics(float& OutAveragePriority, float& OutAverageVariance, int32& OutTotalScenarios) const
{
	OutTotalScenarios = Scenarios.Num();

	if (Scenarios.Num() == 0)
	{
		OutAveragePriority = 0.0f;
		OutAverageVariance = 0.0f;
		return;
	}

	float TotalPrio = 0.0f;
	float TotalVar = 0.0f;

	for (const FMCTSScenarioMetrics& Scenario : Scenarios)
	{
		TotalPrio += Scenario.CalculatePriority();
		TotalVar += Scenario.ValueVariance;
	}

	OutAveragePriority = TotalPrio / Scenarios.Num();
	OutAverageVariance = TotalVar / Scenarios.Num();
}

void UCurriculumManager::PruneOldScenarios(int32 MaxScenarios)
{
	if (Scenarios.Num() <= MaxScenarios)
	{
		return;
	}

	// Keep highest priority scenarios
	TArray<FScenarioPriorityEntry> PriorityEntries;
	PriorityEntries.Reserve(Scenarios.Num());

	for (int32 i = 0; i < Scenarios.Num(); ++i)
	{
		FScenarioPriorityEntry Entry;
		Entry.ScenarioIndex = i;
		Entry.Priority = Scenarios[i].CalculatePriority();
		PriorityEntries.Add(Entry);
	}

	// Sort by priority (descending)
	PriorityEntries.Sort();

	// Keep top MaxScenarios
	TArray<FMCTSScenarioMetrics> NewScenarios;
	NewScenarios.Reserve(MaxScenarios);

	for (int32 i = 0; i < MaxScenarios; ++i)
	{
		NewScenarios.Add(Scenarios[PriorityEntries[i].ScenarioIndex]);
	}

	Scenarios = MoveTemp(NewScenarios);

	// Rebuild priority sum
	RebuildPriorityDistribution();
}

bool UCurriculumManager::ExportToFile(const FString& FilePath) const
{
	TArray<TSharedPtr<FJsonValue>> JsonScenarios;

	for (const FMCTSScenarioMetrics& Scenario : Scenarios)
	{
		TSharedPtr<FJsonObject> JsonScenario = MakeShared<FJsonObject>();

		JsonScenario->SetNumberField(TEXT("CommandType"), Scenario.CommandType);
		JsonScenario->SetNumberField(TEXT("ValueVariance"), Scenario.ValueVariance);
		JsonScenario->SetNumberField(TEXT("PolicyEntropy"), Scenario.PolicyEntropy);
		JsonScenario->SetNumberField(TEXT("VisitCount"), Scenario.VisitCount);
		JsonScenario->SetNumberField(TEXT("AverageValue"), Scenario.AverageValue);
		JsonScenario->SetNumberField(TEXT("ActualReward"), Scenario.ActualReward);
		JsonScenario->SetNumberField(TEXT("Timestamp"), Scenario.Timestamp);
		JsonScenario->SetNumberField(TEXT("Priority"), Scenario.CalculatePriority());

		// Export observation data
		TArray<TSharedPtr<FJsonValue>> ObsArray;
		for (float Value : Scenario.TeamObservation.Data)
		{
			ObsArray.Add(MakeShared<FJsonValueNumber>(Value));
		}
		JsonScenario->SetArrayField(TEXT("Observation"), ObsArray);

		JsonScenarios.Add(MakeShared<FJsonValueObject>(JsonScenario));
	}

	TSharedPtr<FJsonObject> RootObject = MakeShared<FJsonObject>();
	RootObject->SetArrayField(TEXT("Scenarios"), JsonScenarios);
	RootObject->SetNumberField(TEXT("TotalScenarios"), Scenarios.Num());

	FString OutputString;
	TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutputString);

	if (FJsonSerializer::Serialize(RootObject.ToSharedRef(), Writer))
	{
		return FFileHelper::SaveStringToFile(OutputString, *FilePath);
	}

	return false;
}

void UCurriculumManager::RebuildPriorityDistribution()
{
	TotalPriority = 0.0f;

	for (const FMCTSScenarioMetrics& Scenario : Scenarios)
	{
		float Priority = FMath::Pow(Scenario.CalculatePriority(), PrioritizationExponent);
		TotalPriority += Priority;
	}
}
