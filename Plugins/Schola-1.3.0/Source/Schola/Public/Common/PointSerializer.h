// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Common/Points.h"
#include "PointSerializer.generated.h"


/**
 * @brief A class that can serialize a point to a buffer
 * @tparam T The type of buffer to serialize to
 */
template<typename T>
class PointSerializer
{
	/**
	 * @brief Convert a binary point to an intermediate representation
	 * @param[in] Point The point to convert
	 */
	void Visit(FBinaryPoint& Point);

	/**
	 * @brief Convert a discrete point to an intermediate representation
	 * @param[in] Point The point to convert
	 */
	void Visit(FDiscretePoint& Point);

	/**
	 * @brief Convert a box point to an intermediate representation
	 * @param[in] Point The point to convert
	 */
	void Visit(FBoxPoint& Point);

	/**
	 * @brief Convert a dict point to an intermediate representation
	 * @param[in] Point The point to convert
	 */
	void Visit(FDictPoint& Point);

	/**
	 * @brief Serialize the intermediate represnetation to a new buffer
	 * @return A ptr to a new buffer containing the serialized point
	 */
	T* Build();

	/**
	 * @brief Fill an existing buffer from the intermediate representation
	 * @param[out] EmptyBuffer The buffer to fill with the serialized point
	 */
	void Build(T& EmptyBuffer);
};
