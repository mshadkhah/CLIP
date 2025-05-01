// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file
 * @brief Declares the base Solver class that implements common boundary condition logic
 *        and field-level utilities for the CLIP LBM simulation framework.
 *
 * The Solver class provides:
 * - CUDA-ready implementations of physical boundary conditions
 * - Utilities for velocity, Neumann, mirror, and periodic BCs
 * - Shared infrastructure for derived solvers (e.g., NSAllen)
 * - A reusable equilibrium distribution function
 *
 * Derived solvers inherit from this class and implement time stepping and problem-specific logic.
 */

#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <Boundary.cuh>
#include <Domain.cuh>
#include "DataArray.cuh"
#include "Geometry.cuh"

namespace clip
{

    /**
     * @brief Base class providing reusable boundary condition and solver infrastructure for CLIP.
     *
     * This class defines device-ready implementations of all supported boundary conditions,
     * including wall, slip wall, periodic, free-convective, Neumann, and velocity boundaries.
     * It also holds shared simulation structures and CUDA grid information.
     */
    class Solver
    {

    public:
        /**
         * @brief Constructs a Solver and links input, domain, and geometry data.
         * @param idata Simulation input
         * @param domain Domain structure
         * @param DA Reference to simulation field arrays
         * @param boundary Boundary configuration
         * @param geom Embedded geometry
         */
        explicit Solver(const InputData &idata, const Domain &domain, DataArray &DA, const Boundary &boundary, const Geometry &geom);

        /// Destructor
        virtual ~Solver();

        /**
         * @brief Applies periodic boundary conditions.
         * @tparam Q Number of lattice directions
         * @param dev_a Distribution function A
         * @param dev_b Optional: Distribution function B
         */
        template <CLIP_UINT Q>
        void periodicBoundary(CLIP_REAL *dev_a, CLIP_REAL *dev_b = nullptr);

        /**
         * @brief Applies wall (no-slip) boundary conditions.
         * @tparam Q Lattice direction count
         * @tparam dof Number of DOFs per field
         */
        template <CLIP_UINT Q, CLIP_UINT dof>
        void wallBoundary(CLIP_REAL *dev_a, CLIP_REAL *dev_a_post, CLIP_REAL *dev_b = nullptr, CLIP_REAL *dev_b_post = nullptr);

        /**
         * @brief Applies slip wall boundary conditions (tangential velocity preserved).
         * @tparam Q Lattice direction count
         * @tparam dof Number of DOFs per field
         */
        template <CLIP_UINT Q, CLIP_UINT dof>
        void slipWallBoundary(CLIP_REAL *dev_a, CLIP_REAL *dev_a_post, CLIP_REAL *dev_b = nullptr, CLIP_REAL *dev_b_post = nullptr);

        /**
         * @brief Applies free convective (zero normal gradient) boundary conditions.
         * @tparam Q Lattice direction count
         * @tparam dof Number of DOFs per field
         */
        template <CLIP_UINT Q, CLIP_UINT dof>
        void freeConvectBoundary(CLIP_REAL *dev_vel, CLIP_REAL *dev_a, CLIP_REAL *dev_a_post, CLIP_REAL *dev_b = nullptr, CLIP_REAL *dev_b_post = nullptr);

        /**
         * @brief Applies Neumann boundary conditions (zero gradient).
         * @tparam Q Lattice direction count
         * @tparam dof Number of DOFs per field
         */
        template <CLIP_UINT Q, CLIP_UINT dof>
        void NeumannBoundary(CLIP_REAL *dev_a, CLIP_REAL *dev_b = nullptr);

        /**
         * @brief Applies prescribed velocity boundary conditions on inlet/outlet surfaces.
         * @param dev_vel Velocity field
         * @param dev_f Distribution function f
         * @param dev_g Distribution function g
         */
        void velocityBoundary(CLIP_REAL *dev_vel, CLIP_REAL *dev_f, CLIP_REAL *dev_g);

        /**
         * @brief Applies mirror symmetry for scalar fields.
         * @param dev_a Distribution function to mirror
         */
        void mirrorBoundary(CLIP_REAL *dev_a);

        /**
         * @brief Computes the equilibrium distribution function for WMRT-based solvers.
         * @param velSet Lattice velocity set
         * @param q Direction index
         * @param Ux Velocity in x
         * @param Uy Velocity in y
         * @param Uz Velocity in z (optional in 3D)
         * @return Equilibrium distribution value
         */
        __device__ __forceinline__ static CLIP_REAL Equilibrium_new(const WMRT::WMRTvelSet velSet, CLIP_UINT q, CLIP_REAL Ux, CLIP_REAL Uy, CLIP_REAL Uz)
        {
            // using namespace nsAllen;
            const CLIP_INT exq = velSet.ex[q];
            const CLIP_INT eyq = velSet.ey[q];
            const CLIP_REAL waq = velSet.wa[q];

#ifdef ENABLE_2D
            const CLIP_REAL eU = exq * Ux + eyq * Uy;
            const CLIP_REAL U2 = Ux * Ux + Uy * Uy;
#elif defined(ENABLE_3D)
            const CLIP_INT ezq = velSet.ez[q];
            const CLIP_REAL eU = exq * Ux + eyq * Uy + ezq * Uz;
            const CLIP_REAL U2 = Ux * Ux + Uy * Uy + Uz * Uz;
#endif

            return waq * (3.0 * eU + 4.5 * eU * eU - 1.5 * U2);
        }

    private:
    protected:
        const Domain *m_domain;              ///< Pointer to domain geometry
        const InputData *m_idata;            ///< Input configuration
        const Boundary *m_boundary;          ///< Boundary condition data
        const Geometry *m_geom;              ///< Geometry definition
        Geometry::GeometryDevice m_geomPool; ///< Device-side geometry info
        Domain::DomainInfo m_info;           ///< Domain metadata
        Boundary::BCTypeMap m_BCMap;         ///< Boundary condition map
        WMRT::wallBCMap m_wallBCMap;         ///< Wall boundary map (weighted MRT)
        WMRT::slipWallBCMap m_slipWallBCMap; ///< Slip wall map (weighted MRT)
        WMRT::WMRTvelSet m_velSet;           ///< Lattice velocity set
        DataArray *m_DA;                     ///< Data array handler
        dim3 dimGrid, dimBlock;              ///< CUDA kernel launch configuration
    };

}
