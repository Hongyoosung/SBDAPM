// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Tests/AutomationEditorCommon.h"

#include "Common/Points/BinaryPoint.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBinaryPointFromArrayTest, "Schola.Points.BinaryPoint.From Array Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBinaryPointFromArrayTest::RunTest(const FString& Parameters)
{
    TArray<bool> Values = { true, false, true };
    FBinaryPoint BinaryPoint = FBinaryPoint(Values);

    TestEqual(TEXT("BinaryPoint[0] == true"), BinaryPoint[0], true);
    TestEqual(TEXT("BinaryPoint[1] == false"), BinaryPoint[1], false);
    TestEqual(TEXT("BinaryPoint[2] == true"), BinaryPoint[2], true);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBinaryPointAddTest, "Schola.Points.BinaryPoint.Add Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBinaryPointAddTest::RunTest(const FString& Parameters)
{
    FBinaryPoint BinaryPoint = FBinaryPoint();
    BinaryPoint.Add(true);
    BinaryPoint.Add(false);

    TestEqual(TEXT("BinaryPoint[0] == true"), BinaryPoint[0], true);
    TestEqual(TEXT("BinaryPoint[1] == false"), BinaryPoint[1], false);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBinaryPointResetTest, "Schola.Points.BinaryPoint.Reset Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBinaryPointResetTest::RunTest(const FString& Parameters)
{
    FBinaryPoint BinaryPoint = FBinaryPoint();
    BinaryPoint.Add(true);
    BinaryPoint.Add(false);
    BinaryPoint.Reset();

    TestEqual(TEXT("BinaryPoint.Values.Num() == 0"), BinaryPoint.Values.Num(), 0);

    return true;
}