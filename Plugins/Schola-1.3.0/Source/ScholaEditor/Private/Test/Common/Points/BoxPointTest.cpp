// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Tests/AutomationEditorCommon.h"
#include "Common/Points/BoxPoint.h"

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxPointFromArrayTest, "Schola.Points.BoxPoint.From Array Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBoxPointFromArrayTest::RunTest(const FString& Parameters)
{
    TArray<float> Values = { 1.0f, 2.0f, 3.0f };
	FBoxPoint BoxPoint = FBoxPoint(Values.GetData(),3);

    TestEqualExactFloat(TEXT("BoxPoint[0] == 1.0f"), BoxPoint[0], 1.0f);
    TestEqualExactFloat(TEXT("BoxPoint[1] == 2.0f"), BoxPoint[1], 2.0f);
    TestEqualExactFloat(TEXT("BoxPoint[2] == 3.0f"), BoxPoint[2], 3.0f);
    
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxPointAddTest, "Schola.Points.BoxPoint.Add Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBoxPointAddTest::RunTest(const FString& Parameters)
{
    FBoxPoint BoxPoint = FBoxPoint();
    BoxPoint.Add(1.0f);
    BoxPoint.Add(2.0f);

    TestEqualExactFloat(TEXT("BoxPoint[0] == 1.0f"), BoxPoint[0], 1.0f);
    TestEqualExactFloat(TEXT("BoxPoint[1] == 2.0f"), BoxPoint[1], 2.0f);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxPointResetTest, "Schola.Points.BoxPoint.Reset Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBoxPointResetTest::RunTest(const FString& Parameters)
{
    FBoxPoint BoxPoint = FBoxPoint();
    BoxPoint.Add(1.0f);
    BoxPoint.Add(2.0f);
    BoxPoint.Reset();

    TestEqual(TEXT("BoxPoint.Values.Num() == 0"), BoxPoint.Values.Num(), 0);

    return true;
}