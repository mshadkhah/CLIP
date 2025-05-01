// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file
 * @brief Defines the Domain class for managing spatial extents, indexing, and boundary checks
 *        in the CLIP LBM simulation framework.
 *
 * The Domain class computes indexing logic, holds dimensional metadata, and provides
 * utilities for identifying whether a point is inside the domain (with or without ghost cells).
 *
 * It also supports dimensional generality (2D or 3D) and DOF-based flattened indexing
 * for both scalar and vector field layouts.
 */

#pragma once
#include <includes.h>
#include <InputData.cuh>

namespace clip
{

    /**
     * @brief Represents the simulation domain and provides indexing and boundary utility functions.
     *
     * This class stores global indexing info and offers device-safe indexing logic.
     * It supports both ghosted and non-ghosted domain regions, and works for both 2D and 3D setups.
     */
    class Domain
    {

    public:
        /**
         * @brief Construct a Domain object using input parameters.
         * @param idata Reference to input data (e.g., domain dimensions, ghost layers).
         */
        explicit Domain(const InputData &idata);

        /**
         * @brief Holds global domain metadata, including extent and boundaries.
         */
        struct DomainInfo
        {
            CLIP_UINT extent[MAX_DIM];
            CLIP_UINT domainMinIdx[MAX_DIM];
            CLIP_UINT domainMaxIdx[MAX_DIM];
            CLIP_UINT ghostDomainMinIdx[MAX_DIM];
            CLIP_UINT ghostDomainMaxIdx[MAX_DIM];
        };

        /**
         * @brief Computes a flattened 1D index for multi-DOF data.
         *
         * @tparam ndof Number of degrees of freedom (e.g., scalar = 1, vector = 2 or 3).
         * @param domain DomainInfo struct
         * @param i Index in X direction
         * @param j Index in Y direction
         * @param k Index in Z direction
         * @param dof Degree of freedom offset
         * @return Flattened global index for accessing a field
         */
        template <CLIP_UINT ndof = 1>
        __host__ __device__ __forceinline__ static CLIP_UINT getIndex(const DomainInfo &domain, CLIP_UINT i, CLIP_UINT j, CLIP_UINT k, CLIP_UINT dof = SCALAR)
        {

            return ((i * domain.extent[IDX_Y] + j) * domain.extent[IDX_Z] + k) * ndof + dof;
        }

        /**
         * @brief Determines if a node lies within the domain bounds.
         *
         * @tparam dim Spatial dimensionality (2 or 3)
         * @tparam ghosted If true, does not include ghost cell layers
         * @param domain DomainInfo struct
         * @param i Index in X
         * @param j Index in Y
         * @param k Index in Z (optional for 2D)
         * @return true if the node is inside the domain
         */
        template <CLIP_UINT dim, bool ghosted = false>
        __device__ __forceinline__ static bool isInside(const DomainInfo &domain, CLIP_INT i, CLIP_INT j, CLIP_INT k = 0)
        {
            constexpr CLIP_UINT offset = ghosted ? 1 : 0;

            if constexpr (dim == 2)
            {
                return (i >= offset && i < domain.extent[IDX_X] - offset) &&
                       (j >= offset && j < domain.extent[IDX_Y] - offset);
            }
            else if constexpr (dim == 3)
            {
                return (i >= offset && i < domain.extent[IDX_X] - offset) &&
                       (j >= offset && j < domain.extent[IDX_Y] - offset) &&
                       (k >= offset && k < domain.extent[IDX_Z] - offset);
            }
            else
            {
                return false;
            }
        }

        DomainInfo info;      ///< Contains extent and boundary indices
        CLIP_UINT domainSize; ///< Total number of grid points (flattened)
    private:
        CLIP_UINT *m_domainExtent;        ///< Internal extent without ghosts
        CLIP_UINT *m_domainExtentGhosted; ///< Internal extent including ghosts

    protected:
        dim3 dimBlock, dimGrid;   ///< CUDA kernel launch configuration
        const InputData *m_idata; ///< Reference to input parameters
    };

}
