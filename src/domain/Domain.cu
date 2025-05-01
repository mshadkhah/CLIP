// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file
 * @brief Implements the Domain class, which sets up spatial grid extents, indexing,
 *        and ghost region boundaries for 2D and 3D LBM simulations in CLIP.
 *
 * This includes:
 * - Calculation of total and physical domain size
 * - Definition of ghost layers and internal domain bounds
 * - Handling of dimensional specialization (2D vs 3D)
 */

#include <Domain.cuh>

namespace clip
{

    // Constructor: Initializes the computational domain using input parameters
    Domain::Domain(const InputData &idata)
        : m_idata(&idata)
    {

#ifdef ENABLE_2D

        // Set extent (with ghost layers) in X and Y; Z is fixed to 1
        info.extent[IDX_X] = m_idata->params.N[IDX_X] + 2;
        info.extent[IDX_Y] = m_idata->params.N[IDX_Y] + 2;
        info.extent[IDX_Z] = 1;

        // Full domain (including ghost)
        info.ghostDomainMinIdx[IDX_X] = 0;
        info.ghostDomainMinIdx[IDX_Y] = 0;
        info.ghostDomainMinIdx[IDX_Z] = 0;

        info.ghostDomainMaxIdx[IDX_X] = info.extent[IDX_X] - 1;
        info.ghostDomainMaxIdx[IDX_Y] = info.extent[IDX_Y] - 1;
        info.ghostDomainMaxIdx[IDX_Z] = 0;

        // Interior domain (excluding ghost)
        info.domainMinIdx[IDX_X] = 1;
        info.domainMinIdx[IDX_Y] = 1;
        info.domainMinIdx[IDX_Z] = 0;

        info.domainMaxIdx[IDX_X] = info.extent[IDX_X] - 2;
        info.domainMaxIdx[IDX_Y] = info.extent[IDX_Y] - 2;
        info.domainMaxIdx[IDX_Z] = 0;

        // Total size (number of nodes)
        domainSize = info.extent[IDX_X] * info.extent[IDX_Y];

#elif defined(ENABLE_3D)

        // Set extents in X, Y, Z (ghost cells are implicitly included in indexing logic)
        info.extent[IDX_X] = m_idata->params.N[IDX_X];
        info.extent[IDX_Y] = m_idata->params.N[IDX_Y];
        info.extent[IDX_Z] = m_idata->params.N[IDX_Z];

        // Interior domain
        info.domainMinIdx[IDX_X] = 1;
        info.domainMinIdx[IDX_Y] = 1;
        info.domainMinIdx[IDX_Z] = 1;

        info.domainMaxIdx[IDX_X] = info.extent[IDX_X] - 2;
        info.domainMaxIdx[IDX_Y] = info.extent[IDX_Y] - 2;
        info.domainMaxIdx[IDX_Z] = info.extent[IDX_Z] - 2;

        // Ghost domain
        info.ghostDomainMinIdx[IDX_X] = 0;
        info.ghostDomainMinIdx[IDX_Y] = 0;
        info.ghostDomainMinIdx[IDX_Z] = 0;

        info.ghostDomainMaxIdx[IDX_X] = info.extent[IDX_X] - 1;
        info.ghostDomainMaxIdx[IDX_Y] = info.extent[IDX_Y] - 1;
        info.ghostDomainMaxIdx[IDX_Z] = info.extent[IDX_Z] - 1;

        // Total size (including ghost zones)
        domainSize = info.extent[IDX_X] * info.extent[IDX_Y] * info.extent[IDX_Z];
#endif
    };

}