// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Tests/AutomationEditorCommon.h"
#include "Common/Points/DiscretePoint.h"


IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscretePointFromArrayTest, "Schola.Points.DiscretePoint.From Array Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscretePointFromArrayTest::RunTest(const FString& Parameters)
{
    TArray<int> Values = { 1, 2, 3 };
    FDiscretePoint DiscretePoint = FDiscretePoint(Values.GetData(), 3);

    TestEqual(TEXT("DiscretePoint[0] == 1"), DiscretePoint[0], 1);
    TestEqual(TEXT("DiscretePoint[1] == 2"), DiscretePoint[1], 2);
    TestEqual(TEXT("DiscretePoint[2] == 3"), DiscretePoint[2], 3);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscretePointAddTest, "Schola.Points.DiscretePoint.Add Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscretePointAddTest::RunTest(const FString& Parameters)
{
    FDiscretePoint DiscretePoint = FDiscretePoint();
    DiscretePoint.Add(1);
    DiscretePoint.Add(2);

    TestEqual(TEXT("DiscretePoint[0] == 1"), DiscretePoint[0], 1);
    TestEqual(TEXT("DiscretePoint[1] == 2"), DiscretePoint[1], 2);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscretePointResetTest, "Schola.Points.DiscretePoint.Reset Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscretePointResetTest::RunTest(const FString& Parameters)
{
    FDiscretePoint DiscretePoint = FDiscretePoint();
    DiscretePoint.Add(1);
    DiscretePoint.Add(2);
    DiscretePoint.Reset();

    TestEqual(TEXT("DiscretePoint.Values.Num() == 0"), DiscretePoint.Values.Num(), 0);

    return true;
}
