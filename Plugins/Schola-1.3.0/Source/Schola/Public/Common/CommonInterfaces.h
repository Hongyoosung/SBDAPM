// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

/**
 * @brief A singleton template that lets you make a singleton out of a type. Note you can use this on multiple types simultaneously to get multiple singletons.
 * @tparam T The type to make a singleton out of
 */
template <class T>
class Singleton
{
protected:
	static inline T* Instance = nullptr;

public:
	/** 
	 * @brief Does there exist a singleton of this type already?
	 * @return true iff there is a pre-existing instance of this singleton
	*/
	static bool HasInstance()
	{
		return Instance != nullptr;
	}
	/**
	 * @brief Get the typed singleton instance if it exists, otherwise create one
	 * @return A pointer to the singleton instance
	 */
	static T* GetInstance()
	{
		// For this to work the constructor needs to be visible
		// if you are singletoning another normal class, carry on
		// if your singleton inherits you need to declare this a friend class
		if (!Singleton<T>::HasInstance())
		{
			Singleton<T>::Instance = new T();
		}
		return Instance;
	}
};
