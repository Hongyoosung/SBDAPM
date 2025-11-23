// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class ScholaEditor : ModuleRules
{
	public ScholaEditor(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;


		PublicIncludePaths.AddRange(new string[] { });
		PrivateIncludePaths.AddRange(new string[] { "Schola/Private", "ScholaEditor/Private" });
		//TODO figure out if these can be moved to private
		PublicDependencyModuleNames.AddRange(new string[] { "Kismet", "BlueprintEditorLibrary", "Schola" });


		PrivateIncludePathModuleNames.AddRange(new string[] { });
		PrivateDependencyModuleNames.AddRange(new string[] { "Engine", "Core", "BlueprintGraph", "UnrealEd", "CoreUObject" });
		DynamicallyLoadedModuleNames.AddRange(new string[] { });
	}
}
