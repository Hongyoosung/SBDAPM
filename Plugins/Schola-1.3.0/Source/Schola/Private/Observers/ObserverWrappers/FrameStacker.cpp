// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Observers/ObserverWrappers/FrameStacker.h"

FBoxSpace UFrameStacker::WrapBoxObservationSpace(const FBoxSpace& Space)
{
	//Make an allocated FBoxSpace of size equal to MemorySize*Space Size
	this->IndividualSpaceSize = Space.GetFlattenedSize();
	FBoxSpace OutSpace = FBoxSpace();

	if (Space.Shape.Num() > 0)
	{
		OutSpace.Shape = Space.Shape;
		OutSpace.Shape[0] = Space.Shape[0] * this->MemorySize;
		
	}

	for (int i = 0; i < this->MemorySize; i++)
	{
		// Add the dimension to the new space
		OutSpace.Dimensions.Append(Space.Dimensions);
	}

	this->FrameBuffer.Init(this->FillValue, this->GetBufferSize());
	return OutSpace;
}

FBoxPoint UFrameStacker::WrapBoxObservation(const FBoxPoint& Point)
{
	float* DataPtr = this->FrameBuffer.GetData();
	//Move everything back in the array
	for (int i = 0; i < (this->MemorySize - 1); i++)
	{
		FMemory::Memcpy(&DataPtr[i * this->IndividualSpaceSize], &DataPtr[(i + 1) * this->IndividualSpaceSize], this->IndividualSpaceSize * sizeof(float));
	}
	//Copy from the Point to the Buffer
	FMemory::Memcpy(&DataPtr[(this->MemorySize - 1) * this->IndividualSpaceSize], Point.Values.GetData(), this->IndividualSpaceSize * sizeof(float));

	//Create a new box point with the buffer as the observation
	return FBoxPoint(DataPtr, this->MemorySize * this->IndividualSpaceSize);

}

void UFrameStacker::Reset()
{
	// Fill the buffer with the fill value
	FMemory::Memset(this->FrameBuffer.GetData(), this->FillValue, this->GetBufferSize()*sizeof(float));
}

int UFrameStacker::GetBufferSize()
{
	return this->MemorySize * this->IndividualSpaceSize;
}

FString UFrameStacker::GenerateId() const
{
	FString Output = FString("FrameStack");
	//Add the memory size
	Output.Appendf(TEXT("_%d"), this->MemorySize);
	return Output;
}
