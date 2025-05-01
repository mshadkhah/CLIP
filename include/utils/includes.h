// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file
 * @brief Common include header that provides core system libraries, CUDA headers, 
 *        and project-level dependencies for the CLIP framework.
 *
 * This file includes:
 * - C++ standard libraries (I/O, strings, containers, algorithms)
 * - CUDA runtime and device headers
 * - CLIP internal types and logger support
 *
 * It acts as a centralized location to avoid repetitive includes across .cu and .cuh files.
 */


#pragma once


#pragma once

// ----------------- C++ Standard Libraries -----------------
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cstring>
#include <iomanip>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <set>

// ----------------- CUDA Runtime -----------------
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// ----------------- CLIP Project Headers -----------------
#include "DataTypes.cuh"
#include "Logger.cuh"
