// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "../../../Generated/Spaces.pb.h"
#include "Common/Points/PointVariant.h"
#include "Common/IValidatable.h"
#include "Space.generated.h"


using Schola::FundamentalSpace;

/**
 * @brief A class representing a space of possible observations or actions. This is a base class for all spaces.
 */
USTRUCT()
struct SCHOLA_API FSpace
{
	GENERATED_BODY()

	/**
	 * @brief Convert this space to a protobuf message
	 * @return A protobuf message representing this space
	 */
	FundamentalSpace*			   ToProtobuf() const;

	/**
	 * @brief Fill a protobuf message with the data from this space
	 * @param[in] Msg The protobuf message to fill
	 */
	virtual void				   FillProtobuf(FundamentalSpace* Msg) const PURE_VIRTUAL(FSpace::FillProtobuf, return; );

	/**
	 * @brief Get the number of dimensions in this space
	 * @return The number of dimensions in this space
	 */
	virtual int					   GetNumDimensions() const PURE_VIRTUAL(FSpace::FillProtobuf, return 0;);

	/**
	 * @brief Check if this space is empty
	 * @return True if this space is empty, false otherwise
	 */
	virtual bool				   IsEmpty() const PURE_VIRTUAL(FSpace::IsEmpty, return true;);

	/**
	 * @brief Test if an observation is in this space
	 * @param[in] Observation The observation to validate
	 * @return An enum indicating the result of the validation
	 */
	virtual ESpaceValidationResult Validate(TPoint& Observation) const PURE_VIRTUAL(FSpace::IsEmpty, return ESpaceValidationResult::NoResults;);
	
	/**
	 * @brief Get the size of the flattened representation of this space
	 * @return The size of the flattened representation of this space
	 */
	virtual int					   GetFlattenedSize() const PURE_VIRTUAL(FSpace::GetFlattenedSize, return 0;);

	/**
	 * @brief Create a TPoint from this space
	 * @return A TPoint belonging to this space, with correctly set variant type.
	 */
	virtual TPoint				   MakeTPoint() const PURE_VIRTUAL(FSpace::MakeTPoint, return TPoint(););
	
	/**
	 * @brief Unflatten an action from a buffer
	 * @param[in] Data The buffer to unflatten from
	 * @param[in] Offset The offset into the buffer to start unflattening from
	 */
	virtual TPoint				   UnflattenAction(const TArray<float>& Data, int Offset = 0) const PURE_VIRTUAL(FSpace::UnflattenAction, return TPoint(););
	/**
	 * @brief Flatten a point into a buffer
	 * @param[in,out] Buffer The buffer to flatten into
	 * @param[in] Point The point to flatten
	 */
	virtual void				   FlattenPoint(TArrayView<float> Buffer, const TPoint& Point) const PURE_VIRTUAL(FSpace::FlattenPoint, return; );
	virtual ~FSpace() = default;
	
};