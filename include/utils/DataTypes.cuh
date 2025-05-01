// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file
 * @brief Core type definitions and macros for precision control, dimensionality, and indexing in CLIP.
 *
 * This header provides:
 * - Unified typedefs for scalar types (`CLIP_REAL`, `CLIP_UINT`, `CLIP_INT`)
 * - Precision control through `USE_SINGLE_PRECISION`
 * - Dimensional configuration using `ENABLE_2D` or `ENABLE_3D`
 * - Indexing constants (`IDX_X`, `IDX_Y`, `IDX_Z`) and scalar field base
 * - Safe GPU memory deallocation macro
 *
 * All CLIP simulation headers rely on these core definitions for portability and consistency.
 */

#pragma once


/// Unsigned integer type for indexing
typedef unsigned int CLIP_UINT;

/// Signed integer type
typedef int CLIP_INT;

/// Real number type (float or double based on precision)
#ifdef USE_SINGLE_PRECISION
typedef float CLIP_REAL;  ///< Use single-precision float
#else
typedef double CLIP_REAL; ///< Use double-precision float
#endif

// ---------------------- Dimensionality Macros ----------------------

#ifdef ENABLE_2D
#define DIM 2         ///< Spatial dimension = 2
#elif defined(ENABLE_3D)
#define DIM 3         ///< Spatial dimension = 3
#endif

#define MAX_DIM 3         ///< Maximum number of spatial dimensions
#define SCALAR_FIELD 1    ///< Default number of DOFs for scalar fields

// ---------------------- Index Constants ----------------------

#define SCALAR 0     ///< Scalar field index
#define IDX_X 0      ///< X dimension index
#define IDX_Y 1      ///< Y dimension index
#define IDX_Z 2      ///< Z dimension index

// ---------------------- Safe CUDA Free ----------------------

/**
 * @brief Safely deallocates GPU memory and nullifies pointer.
 */
#define SAFE_CUDA_FREE(ptr) \
    if (ptr)                \
    {                       \
        cudaFree(ptr);      \
        ptr = nullptr;      \
    }
