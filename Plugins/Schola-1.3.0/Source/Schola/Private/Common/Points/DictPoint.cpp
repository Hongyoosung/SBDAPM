// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Common/Points/DictPoint.h"

void FDictPoint::Reset()
{
	this->Points.Reset(this->Points.Num());
}
