// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Tests/AutomationEditorCommon.h"

#include "Common/Spaces/BinarySpace.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBinarySpaceDefaultTest, "Schola.Spaces.BinarySpace.Default Creation Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBinarySpaceDefaultTest::RunTest(const FString& Parameters)
{
    FBinarySpace BinarySpace = FBinarySpace();

    TestEqual(TEXT("BinarySpace.Shape == 0"), BinarySpace.Shape, 0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBinarySpaceShapeTest, "Schola.Spaces.BinarySpace.Shape Creation Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBinarySpaceShapeTest::RunTest(const FString& Parameters)
{
    FBinarySpace BinarySpace = FBinarySpace(10);

    TestEqual(TEXT("BinarySpace.Shape == 10"), BinarySpace.Shape, 10);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBinarySpaceMergeTest, "Schola.Spaces.BinarySpace.Merge Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBinarySpaceMergeTest::RunTest(const FString& Parameters)
{
    FBinarySpace BinarySpace = FBinarySpace(10);
    FBinarySpace OtherBinarySpace = FBinarySpace(5);

    BinarySpace.Merge(OtherBinarySpace);

    TestEqual(TEXT("BinarySpace.Shape == 15"), BinarySpace.Shape, 15);

    return true;
}