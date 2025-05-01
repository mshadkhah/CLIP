// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file
 * @brief Defines the NSAllen class, a Navier–Stokes solver using the Allen–Cahn
 *        phase-field approach within the CLIP LBM framework.
 *
 * This class extends the generic Solver interface and implements:
 * - Allen–Cahn-based multiphase modeling
 * - Velocity and volume fraction computation
 * - Specialized streaming and collision logic
 * - Boundary condition application (periodic, wall, convective, Neumann)
 *
 * It operates on the GPU using CUDA and manages interaction with geometry,
 * boundary conditions, and field memory (via DataArray).
 */

#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <Solver.cuh>
#include <DataArray.cuh>
#include "WMRT.cuh"
#include "Geometry.cuh"

namespace clip
{

    /**
     * @brief Implements the Allen–Cahn-based Navier–Stokes solver using LBM.
     *
     * The NSAllen class integrates the Allen–Cahn phase field model with the Lattice Boltzmann Method (LBM)
     * for simulating interfacial flows. It derives from the base Solver class and performs domain setup,
     * equilibrium computation, volume fraction tracking, and application of custom boundary conditions.
     */
    class NSAllen : public Solver
    {

    public:
        /**
         * @brief Constructor for NSAllen solver.
         * @param idata Simulation input configuration
         * @param domain Domain geometry and extent
         * @param DA Field data manager
         * @param boundary Boundary condition object
         * @param geom Geometry manager object
         */
        explicit NSAllen(const InputData &idata, const Domain &domain, DataArray &DA, const Boundary &boundary, const Geometry &geom);

        /// Destructor
        ~NSAllen();

        /**
         * @brief Computes equilibrium distribution for a given velocity set.
         * @param velSet Lattice velocity set
         * @param q Direction index
         * @param Ux X-velocity
         * @param Uy Y-velocity
         * @param Uz Z-velocity (optional, default = 0)
         * @return Equilibrium distribution value
         */
        __device__ __forceinline__ static CLIP_REAL Equilibrium_new(const WMRT::WMRTvelSet velSet, CLIP_UINT q, CLIP_REAL Ux, CLIP_REAL Uy, CLIP_REAL Uz = 0);

        /**
         * @brief Calculates volume fraction force (VF) terms.
         *
         * @tparam q Number of lattice directions
         * @tparam dim Number of spatial dimensions
         * @param velSet Lattice velocity set
         * @param params Simulation parameters
         * @param gneq Output: nonequilibrium components
         * @param fv Output: force terms in each direction
         * @param tau Relaxation time
         * @param dcdx Phase field gradient (X)
         * @param dcdy Phase field gradient (Y)
         * @param dcdz Phase field gradient (Z, optional)
         */
        template <CLIP_UINT q, size_t dim>
        __device__ __forceinline__ static void calculateVF(const WMRT::WMRTvelSet velSet, const InputData::SimParams params, CLIP_REAL gneq[q], CLIP_REAL fv[dim], CLIP_REAL tau, CLIP_REAL dcdx, CLIP_REAL dcdy, CLIP_REAL dcdz = 0);

        struct inletGeom; ///< (Optional) Placeholder or user-defined inlet structure

        /// Main time loop or solution iteration
        void solve();

        /// Computes macroscopic quantities (rho, u) from distribution functions
        void macroscopic();

        /// Sets the initial state of the simulation
        void initialCondition();

        /// Initializes device memory and data structures
        void deviceInitializer();

    private:
        /// Internal initializer for field arrays and simulation state
        void initialization();

        WMRT::WMRTvelSet m_velset;           ///< Lattice velocity set
        InputData::SimParams m_params;       ///< Simulation physical parameters
        const Boundary *m_boundary;          ///< Pointer to boundary conditions
        const Geometry *m_geom;              ///< Pointer to geometry object
        Geometry::GeometryDevice m_geomPool; ///< Device-side geometry representation
        Domain::DomainInfo m_info;           ///< Domain information struct
        dim3 dimGrid, dimBlock;              ///< CUDA grid/block configuration

        /// Performs LBM streaming step
        void streaming();

        /// Performs LBM collision step
        void collision();

        /// Applies periodic boundary conditions
        void applyPeriodicBoundary();

        /// Applies wall boundary conditions
        void applyWallBoundary();

        /// Applies free convective boundary conditions
        void applyFreeConvectBoundary();

        /// Applies Neumann (zero-gradient) boundary conditions
        void applyNeumannBoundary();
    };
}