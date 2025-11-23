// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Common/Spaces/Space.h"

FundamentalSpace* FSpace::ToProtobuf() const
{
	FundamentalSpace* Msg = new FundamentalSpace();
	this->FillProtobuf(Msg);
	return Msg;
}
