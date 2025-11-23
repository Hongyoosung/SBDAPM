// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Tests/AutomationEditorCommon.h"
#include "Common/Points/DictPoint.h"

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictPointAddExistingPointTest, "Schola.Points.DictPoint.Add Existing Point Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDictPointAddExistingPointTest::RunTest(const FString& Parameters)
{
    FDictPoint DictPoint = FDictPoint();
    FBoxPoint BoxPoint = FBoxPoint();
    BoxPoint.Add(1.0f);
    BoxPoint.Add(2.0f);
    TPoint Point;
    Point.Set<FBoxPoint>(BoxPoint);
    DictPoint.Add(Point);

    TestEqualExactFloat(TEXT("DictPoint[0].Get<FBoxPoint>()[0] == 1.0f"), DictPoint[0].Get<FBoxPoint>()[0], 1.0f);
	TestEqualExactFloat(TEXT("DictPoint[0].Get<FBoxPoint>()[1]"), DictPoint[0].Get<FBoxPoint>()[1], 2.0f);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictPointEmplacePointTest, "Schola.Points.DictPoint.Emplace Point Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDictPointEmplacePointTest::RunTest(const FString& Parameters)
{
    FDictPoint DictPoint = FDictPoint();
    TPoint& Point = DictPoint.Add();
    Point.Set<FBoxPoint>(FBoxPoint());

    TestEqual(TEXT("DictPoint[0].Values.Num() == 0"), DictPoint[0].Get<FBoxPoint>().Values.Num(), 0);
    
    return true;
}

