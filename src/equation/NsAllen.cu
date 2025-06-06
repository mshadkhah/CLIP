// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file NsAllen.cu
 * @brief Implementation of a two-phase Lattice Boltzmann Method (LBM) solver using the Allen–Cahn phase-field model.
 *
 * This file contains the CUDA kernels and host-side logic for simulating interfacial flows
 * such as bubbles, drops, jets, and Rayleigh–Taylor instability using the weighted MRT collision operator.
 *
 * @author Mehdi
 * @date 2025
 *
 * @details
 * The solver supports:
 * - Phase-field Allen–Cahn interface tracking
 * - Gravity and capillary forces
 * - Variable viscosity and density
 * - Free-convective, wall, slip, Neumann, and periodic boundaries
 * - MRT-based force coupling
 * - Conservative mass-preserving Allen–Cahn scheme
 *
 * Key Components:
 * - `initialCondition()` initializes `c` based on geometry.
 * - `deviceInitializer()` prepares initial fields like `rho`, `mu`, `vel`, etc.
 * - `solve()` runs one LBM time step with streaming, collision, and BCs.
 * - `collision()`, `streaming()`, and `macroscopic()` implement core solver steps.
 * - CUDA kernels implement equilibrium computation, force coupling, gradients, and chemical potential.
 *
 * Assumes usage of compile-time macros: `ENABLE_2D` or `ENABLE_3D`, and user-defined `InputData`, `Boundary`, `Geometry`, `Domain`, and `DataArray` classes.
 */

#include <NsAllen.cuh>
#include <Solver.cuh>

namespace clip
{

    /// Initializes fields and prepares velocity set, grid/block dimensions, and geometry pool.
    ///  idata Simulation input data
    ///  domain Computational domain
    ///  DA Data storage object (host/device)
    ///  boundary Boundary condition handler
    ///  geom Geometry object defining physical shapes (via SDF)

    NSAllen::NSAllen(const InputData &idata, const Domain &domain, DataArray &DA, const Boundary &boundary, const Geometry &geom)
        : Solver(idata, domain, DA, boundary, geom), m_boundary(&boundary), m_geom(&geom)
    {

        m_info = m_domain->info;
        m_params = m_idata->params;
        dimGrid = m_DA->dimGrid;
        dimBlock = m_DA->dimBlock;
        m_geomPool = m_geom->getDeviceStruct();
    };

    /**
     * @brief Initializes host-side phase field variable `c` based on geometry and case type.
     */
    void NSAllen::initialization()
    {
    }

    NSAllen::~NSAllen()
    {
    }

    /// Computes equilibrium distribution for a given direction and velocity.
    ///  velSet Lattice velocity set
    ///  q Direction index
    ///  Ux Local velocity in x-direction
    ///  Uy Local velocity in y-direction
    ///  Uz Local velocity in z-direction
    ///  Equilibrium value for direction q

    __device__ __forceinline__ CLIP_REAL NSAllen::Equilibrium_new(const WMRT::WMRTvelSet velSet, CLIP_UINT q, CLIP_REAL Ux, CLIP_REAL Uy, CLIP_REAL Uz)
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

    /// Calculates body force contributions from non-equilibrium parts of `f`.
    ///  q Number of directions
    ///  dim Spatial dimension
    ///  gneq Non-equilibrium components
    ///  fv Output force vector

    template <CLIP_UINT q, size_t dim>
    __device__ __forceinline__ void NSAllen::calculateVF(const WMRT::WMRTvelSet velSet, const InputData::SimParams params, CLIP_REAL gneq[q], CLIP_REAL fv[dim], CLIP_REAL tau, CLIP_REAL dcdx, CLIP_REAL dcdy, CLIP_REAL dcdz)
    {
        CLIP_REAL sxx = 0;
        CLIP_REAL sxy = 0;
        CLIP_REAL syy = 0;
        CLIP_REAL szy = 0;
        CLIP_REAL szx = 0;
#ifdef ENABLE_3D
        CLIP_REAL szz = 0;
#endif
        const CLIP_REAL rhoDiff = params.RhoH - params.RhoL;

#pragma unroll
        for (int i = 0; i < q; i++)
        {
            sxx += velSet.ex[i] * velSet.ex[i] * gneq[i];
            sxy += velSet.ex[i] * velSet.ey[i] * gneq[i];
            syy += velSet.ey[i] * velSet.ey[i] * gneq[i];
#ifdef ENABLE_3D
            szy += velSet.ez[i] * velSet.ey[i] * gneq[i];
            szx += velSet.ez[i] * velSet.ex[i] * gneq[i];
            szz += velSet.ez[i] * velSet.ez[i] * gneq[i];
#endif
        }
        fv[IDX_X] = -tau * (sxx * dcdx + sxy * dcdy + szx * dcdz) * rhoDiff;
        fv[IDX_Y] = -tau * (sxy * dcdx + syy * dcdy + szy * dcdz) * rhoDiff;
#ifdef ENABLE_3D
        fv[IDX_Z] = -tau * (szz * dcdz + szx * dcdx + szy * dcdy) * rhoDiff;
#endif
    }

    /**
     * @brief Initializes device-side distributions, density, pressure, velocity, and normals.
     */
    __global__ void KernelInitializeDistributions(const WMRT::WMRTvelSet velSet, const InputData::SimParams params, const Domain::DomainInfo domain, CLIP_REAL *dev_f, CLIP_REAL *dev_g, CLIP_REAL *dev_f_post, CLIP_REAL *dev_g_post,
                                                  CLIP_REAL *dev_c, CLIP_REAL *dev_rho, CLIP_REAL *dev_p, CLIP_REAL *dev_vel, CLIP_REAL *dev_normal)
    {
        constexpr CLIP_UINT Q = WMRT::WMRTvelSet::Q;
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = Domain::getIndex(domain, i, j, k);
        const CLIP_UINT idx_X = Domain::getIndex<DIM>(domain, i, j, k, IDX_X);
        const CLIP_UINT idx_Y = Domain::getIndex<DIM>(domain, i, j, k, IDX_Y);

#ifdef ENABLE_3D
        const CLIP_UINT idx_Z = Domain::getIndex<DIM>(domain, i, j, k, IDX_Z);
#endif

        if (Domain::isInside<DIM>(domain, i, j, k))
        {
            dev_rho[idx_SCALAR] = (params.RhoL) + dev_c[idx_SCALAR] * ((params.RhoH) - (params.RhoL));

            if (params.caseType == InputData::CaseType::Bubble)
            {
                dev_p[idx_SCALAR] = dev_p[idx_SCALAR] - dev_c[idx_SCALAR] * params.sigma / (params.referenceLength / 2.0) / (dev_rho[idx_SCALAR] / 3.0);
            }
            else if (params.caseType == InputData::CaseType::Drop)
            {
                dev_p[idx_SCALAR] = dev_p[idx_SCALAR] + dev_c[idx_SCALAR] * params.sigma / (params.referenceLength / 2.0) / (dev_rho[idx_SCALAR] / 3.0);
            }
            else
            {
                dev_p[idx_SCALAR] = 0;
            }

            dev_vel[idx_X] = 0;
            dev_vel[idx_Y] = 0;

#ifdef ENABLE_3D
            dev_vel[idx_Z] = 0;
#endif

#pragma unroll
            for (CLIP_UINT q = 0; q < Q; q++)
            {

#ifdef ENABLE_2D
                const CLIP_REAL ga_wa = NSAllen::Equilibrium_new(velSet, q, dev_vel[idx_X], dev_vel[idx_Y]);

                const CLIP_REAL hlp = velSet.wa[q] * ((1.0 - 4.0 * ((dev_c[idx_SCALAR] - 0.50) * (dev_c[idx_SCALAR] - 0.50))) /
                                                      params.interfaceWidth * (velSet.ex[q] * dev_normal[idx_X] + velSet.ey[q] * dev_normal[idx_Y]));

#elif defined(ENABLE_3D)
                const CLIP_REAL ga_wa = NSAllen::Equilibrium_new(velSet, q, dev_vel[idx_X], dev_vel[idx_Y], dev_vel[idx_Z]);
                const CLIP_REAL hlp = velSet.wa[q] * ((1.0 - 4.0 * ((dev_c[idx_SCALAR] - 0.50) * (dev_c[idx_SCALAR] - 0.50))) / params.interfaceWidth * (velSet.ex[q] * dev_normal[idx_X] + velSet.ey[q] * dev_normal[idx_Y] + velSet.ez[q] * dev_normal[idx_Z]));
#endif

                const CLIP_REAL Gamma = ga_wa + velSet.wa[q];

                //*******************geq
                dev_g_post[Domain::getIndex<Q>(domain, i, j, k, q)] = dev_c[idx_SCALAR] * Gamma - 0.50 * hlp;
                dev_g[Domain::getIndex<Q>(domain, i, j, k, q)] = dev_c[idx_SCALAR] * Gamma - 0.50 * hlp;
                //*******************geq
                dev_f_post[Domain::getIndex<Q>(domain, i, j, k, q)] = dev_p[idx_SCALAR] * velSet.wa[q] + ga_wa;
                dev_f[Domain::getIndex<Q>(domain, i, j, k, q)] = dev_p[idx_SCALAR] * velSet.wa[q] + ga_wa;
            }
        }
    }

    /**
     * @brief Computes chemical potential from current phase field using Allen–Cahn model.
     */
    __global__ void Chemical_Potential(const InputData::SimParams params, const Domain::DomainInfo domain, double *dev_c, double *dev_mu)
    {

        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = Domain::getIndex(domain, i, j, k);

        if (Domain::isInside<DIM, true>(domain, i, j, k))
        {

#ifdef ENABLE_2D
            const CLIP_REAL D2C = (dev_c[Domain::getIndex(domain, i - 1, j - 1, 0)] + dev_c[Domain::getIndex(domain, i + 1, j - 1, 0)] + dev_c[Domain::getIndex(domain, i - 1, j + 1, 0)] +
                                   dev_c[Domain::getIndex(domain, i + 1, j + 1, 0)] + 4.0 * (dev_c[Domain::getIndex(domain, i, j - 1, 0)] + dev_c[Domain::getIndex(domain, i - 1, j, 0)] + dev_c[Domain::getIndex(domain, i + 1, j, 0)] + dev_c[Domain::getIndex(domain, i, j + 1, 0)]) - 20 * dev_c[Domain::getIndex(domain, i, j, 0)]) /
                                  6.0;
#elif defined(ENABLE_3D)

            const CLIP_REAL D2C = (20.0 * (dev_c[Domain::getIndex(domain, i + 1, j, k)] + dev_c[Domain::getIndex(domain, i - 1, j, k)] + dev_c[Domain::getIndex(domain, i, j + 1, k)] + dev_c[Domain::getIndex(domain, i, j - 1, k)] + dev_c[Domain::getIndex(domain, i, j, k + 1)] + dev_c[Domain::getIndex(domain, i, j, k - 1)]) +
                                   6.0 * (dev_c[Domain::getIndex(domain, i + 1, j + 1, k)] + dev_c[Domain::getIndex(domain, i, j + 1, k - 1)] + dev_c[Domain::getIndex(domain, i - 1, j + 1, k)] + dev_c[Domain::getIndex(domain, i, j + 1, k + 1)] + dev_c[Domain::getIndex(domain, i + 1, j, k + 1)] + dev_c[Domain::getIndex(domain, i + 1, j, k - 1)] +
                                          dev_c[Domain::getIndex(domain, i - 1, j, k - 1)] + dev_c[Domain::getIndex(domain, i - 1, j, k + 1)] + dev_c[Domain::getIndex(domain, i, j - 1, k + 1)] + dev_c[Domain::getIndex(domain, i + 1, j - 1, k)] + dev_c[Domain::getIndex(domain, i, j - 1, k - 1)] + dev_c[Domain::getIndex(domain, i - 1, j - 1, k)]) +
                                   (dev_c[Domain::getIndex(domain, i + 1, j + 1, k + 1)] + dev_c[Domain::getIndex(domain, i + 1, j + 1, k - 1)] + dev_c[Domain::getIndex(domain, i - 1, j + 1, k - 1)] + dev_c[Domain::getIndex(domain, i - 1, j + 1, k + 1)] + dev_c[Domain::getIndex(domain, i + 1, j - 1, k + 1)] + dev_c[Domain::getIndex(domain, i + 1, j - 1, k - 1)] +
                                    dev_c[Domain::getIndex(domain, i - 1, j - 1, k - 1)] + dev_c[Domain::getIndex(domain, i - 1, j - 1, k + 1)]) -
                                   200.0 * dev_c[Domain::getIndex(domain, i, j, k)]) /
                                  48.0;
#endif

            dev_mu[idx_SCALAR] = 4.0 * params.betaConstant * dev_c[idx_SCALAR] * (dev_c[idx_SCALAR] - 1.0) * (dev_c[idx_SCALAR] - 0.50) - params.kConstant * D2C;
        }
    }

    /**
     * @brief Computes normalized interface normals using finite difference of `c`.
     */
    __global__ void normal_FD(const Domain::DomainInfo domain, CLIP_REAL *dev_dc, CLIP_REAL *dev_normal)
    {

        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = Domain::getIndex(domain, i, j, k);
        const CLIP_UINT idx_X = Domain::getIndex<DIM>(domain, i, j, k, IDX_X);
        const CLIP_UINT idx_Y = Domain::getIndex<DIM>(domain, i, j, k, IDX_Y);

#ifdef ENABLE_3D
        const CLIP_UINT idx_Z = Domain::getIndex<DIM>(domain, i, j, k, IDX_Z);
#endif

        if (Domain::isInside<DIM, true>(domain, i, j, k))
        {

#ifdef ENABLE_2D

            const CLIP_REAL tmp = sqrt((dev_dc[idx_X] * dev_dc[idx_X]) + (dev_dc[idx_Y] * dev_dc[idx_Y])) + 1e-32;

            dev_normal[idx_X] = dev_dc[idx_X] / tmp;
            dev_normal[idx_Y] = dev_dc[idx_Y] / tmp;

#elif defined(ENABLE_3D)

            const CLIP_REAL tmp = sqrt((dev_dc[idx_X] * dev_dc[idx_X]) + (dev_dc[idx_Y] * dev_dc[idx_Y]) + (dev_dc[idx_Z] * dev_dc[idx_Z])) + 1e-32;

            dev_normal[idx_X] = dev_dc[idx_X] / tmp;
            dev_normal[idx_Y] = dev_dc[idx_Y] / tmp;
            dev_normal[idx_Z] = dev_dc[idx_Z] / tmp;
#endif
        }
    }

    /**
     * @brief Computes isotropic central difference gradient of phase field `c`.
     */
    __global__ void Isotropic_Gradient(const Domain::DomainInfo domain, CLIP_REAL *dev_c, CLIP_REAL *dev_dc)
    {

        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = Domain::getIndex(domain, i, j, k);
        const CLIP_UINT idx_X = Domain::getIndex<DIM>(domain, i, j, k, IDX_X);
        const CLIP_UINT idx_Y = Domain::getIndex<DIM>(domain, i, j, k, IDX_Y);

#ifdef ENABLE_3D
        const CLIP_UINT idx_Z = Domain::getIndex<DIM>(domain, i, j, k, IDX_Z);
#endif

        if (Domain::isInside<DIM, true>(domain, i, j, k))
        {

#ifdef ENABLE_2D

            dev_dc[idx_X] = (dev_c[Domain::getIndex(domain, i + 1, j, k)] - dev_c[Domain::getIndex(domain, i - 1, j, k)]) / 3.0 +
                            (dev_c[Domain::getIndex(domain, i + 1, j - 1, k)] + dev_c[Domain::getIndex(domain, i + 1, j + 1, k)] - dev_c[Domain::getIndex(domain, i - 1, j - 1, k)] - dev_c[Domain::getIndex(domain, i - 1, j + 1, k)]) / 12.0;

            dev_dc[idx_Y] = (dev_c[Domain::getIndex(domain, i, j + 1, k)] - dev_c[Domain::getIndex(domain, i, j - 1, k)]) / 3.0 +
                            (dev_c[Domain::getIndex(domain, i - 1, j + 1, k)] + dev_c[Domain::getIndex(domain, i + 1, j + 1, k)] - dev_c[Domain::getIndex(domain, i - 1, j - 1, k)] - dev_c[Domain::getIndex(domain, i + 1, j - 1, k)]) / 12.0;

#elif defined(ENABLE_3D)

            dev_dc[idx_X] = (0.50) * ((4.0 / 9.0) * (dev_c[Domain::getIndex(domain, i + 1, j, k)] - dev_c[Domain::getIndex(domain, i - 1, j, k)]) + (1.0 / 9.0) * ((dev_c[Domain::getIndex(domain, i + 1, j, k + 1)] + dev_c[Domain::getIndex(domain, i + 1, j, k - 1)] + dev_c[Domain::getIndex(domain, i + 1, j + 1, k)] + dev_c[Domain::getIndex(domain, i + 1, j - 1, k)]) - (dev_c[Domain::getIndex(domain, i - 1, j, k + 1)] + dev_c[Domain::getIndex(domain, i - 1, j, k - 1)] + dev_c[Domain::getIndex(domain, i - 1, j + 1, k)] + dev_c[Domain::getIndex(domain, i - 1, j - 1, k)])) +
                                      (1.0 / 36.0) * ((dev_c[Domain::getIndex(domain, i + 1, j + 1, k + 1)] + dev_c[Domain::getIndex(domain, i + 1, j - 1, k + 1)] + dev_c[Domain::getIndex(domain, i + 1, j + 1, k - 1)] + dev_c[Domain::getIndex(domain, i + 1, j - 1, k - 1)]) -
                                                      (dev_c[Domain::getIndex(domain, i - 1, j + 1, k + 1)] + dev_c[Domain::getIndex(domain, i - 1, j - 1, k + 1)] + dev_c[Domain::getIndex(domain, i - 1, j + 1, k - 1)] + dev_c[Domain::getIndex(domain, i - 1, j - 1, k - 1)])));

            dev_dc[idx_Y] = (0.50) * ((4.0 / 9.0) * (dev_c[Domain::getIndex(domain, i, j + 1, k)] - dev_c[Domain::getIndex(domain, i, j - 1, k)]) + (1.0 / 9.0) * ((dev_c[Domain::getIndex(domain, i + 1, j + 1, k)] + dev_c[Domain::getIndex(domain, i - 1, j + 1, k)] + dev_c[Domain::getIndex(domain, i, j + 1, k + 1)] + dev_c[Domain::getIndex(domain, i, j + 1, k - 1)]) - (dev_c[Domain::getIndex(domain, i, j - 1, k + 1)] + dev_c[Domain::getIndex(domain, i, j - 1, k - 1)] + dev_c[Domain::getIndex(domain, i + 1, j - 1, k)] + dev_c[Domain::getIndex(domain, i - 1, j - 1, k)])) +
                                      (1.0 / 36.0) * ((dev_c[Domain::getIndex(domain, i + 1, j + 1, k + 1)] + dev_c[Domain::getIndex(domain, i - 1, j + 1, k + 1)] + dev_c[Domain::getIndex(domain, i + 1, j + 1, k - 1)] + dev_c[Domain::getIndex(domain, i - 1, j + 1, k - 1)]) -
                                                      (dev_c[Domain::getIndex(domain, i + 1, j - 1, k + 1)] + dev_c[Domain::getIndex(domain, i - 1, j - 1, k + 1)] + dev_c[Domain::getIndex(domain, i + 1, j - 1, k - 1)] + dev_c[Domain::getIndex(domain, i - 1, j - 1, k - 1)])));

            dev_dc[idx_Z] = (0.50) * ((4.0 / 9.0) * (dev_c[Domain::getIndex(domain, i, j, k + 1)] - dev_c[Domain::getIndex(domain, i, j, k - 1)]) + (1.0 / 9.0) * ((dev_c[Domain::getIndex(domain, i + 1, j, k + 1)] + dev_c[Domain::getIndex(domain, i - 1, j, k + 1)] + dev_c[Domain::getIndex(domain, i, j + 1, k + 1)] + dev_c[Domain::getIndex(domain, i, j - 1, k + 1)]) - (dev_c[Domain::getIndex(domain, i + 1, j, k - 1)] + dev_c[Domain::getIndex(domain, i - 1, j, k - 1)] + dev_c[Domain::getIndex(domain, i, j + 1, k - 1)] + dev_c[Domain::getIndex(domain, i, j - 1, k - 1)])) +
                                      (1.0 / 36.0) * ((dev_c[Domain::getIndex(domain, i + 1, j + 1, k + 1)] + dev_c[Domain::getIndex(domain, i + 1, j - 1, k + 1)] + dev_c[Domain::getIndex(domain, i - 1, j + 1, k + 1)] + dev_c[Domain::getIndex(domain, i - 1, j - 1, k + 1)]) -
                                                      (dev_c[Domain::getIndex(domain, i + 1, j + 1, k - 1)] + dev_c[Domain::getIndex(domain, i - 1, j + 1, k - 1)] + dev_c[Domain::getIndex(domain, i + 1, j - 1, k - 1)] + dev_c[Domain::getIndex(domain, i - 1, j - 1, k - 1)])));

#endif
        }
    }

    /**
     * @brief Streams lattice values from neighboring nodes to current node for `f` or `g`.
     */
    __global__ void kernelStreaming(const WMRT::WMRTvelSet velSet, const Domain::DomainInfo domain, CLIP_REAL *f, CLIP_REAL *f_post)
    {

        constexpr CLIP_UINT Q = WMRT::WMRTvelSet::Q;
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = Domain::getIndex(domain, i, j, k);

        if (Domain::isInside<DIM, true>(domain, i, j, k))
        {
#pragma unroll
            for (int q = 0; q < Q; q++)
            {

#ifdef ENABLE_3D

#endif

#ifdef ENABLE_2D
                const CLIP_UINT id = i - velSet.ex[q];
                const CLIP_UINT jd = j - velSet.ey[q];
                const CLIP_UINT kd = 0;
#elif defined(ENABLE_3D)
                const CLIP_UINT id = i - velSet.ex[q];
                const CLIP_UINT jd = j - velSet.ey[q];
                const CLIP_UINT kd = k - velSet.ez[q];
#endif

                {
                    f_post[Domain::getIndex<Q>(domain, i, j, k, q)] = f[Domain::getIndex<Q>(domain, id, jd, kd, q)];
                    // if (i == 16 && j == 1)
                    // printf ("q = %d  f = %f\n", q, f_post[Domain::getIndex<Q>(domain, i, j, k, q)]);
                }
            }
        }
    }

    /**
     * @brief Computes macroscopic velocity and pressure from `f_post`, including body forces.
     */
    __global__ void kernelMacroscopicf(const WMRT::WMRTvelSet velSet, const InputData::SimParams params, const Domain::DomainInfo domain,
                                       CLIP_REAL *dev_p, CLIP_REAL *dev_rho, CLIP_REAL *dev_c, CLIP_REAL *dev_f_post, CLIP_REAL *dev_dc, CLIP_REAL *dev_vel, CLIP_REAL *dev_mu)
    {

        constexpr CLIP_UINT Q = WMRT::WMRTvelSet::Q;
        CLIP_REAL gneq[Q], tmp[Q], fv[DIM], fg[DIM], tau;
        const CLIP_REAL drho3 = (params.RhoH - params.RhoL) / 3.0;
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = Domain::getIndex(domain, i, j, k);
        const CLIP_UINT idx_X = Domain::getIndex<DIM>(domain, i, j, k, IDX_X);
        const CLIP_UINT idx_Y = Domain::getIndex<DIM>(domain, i, j, k, IDX_Y);

#ifdef ENABLE_3D
        const CLIP_UINT idx_Z = Domain::getIndex<DIM>(domain, i, j, k, IDX_Z);
#endif

        if (Domain::isInside<DIM, true>(domain, i, j, k))
        {

            dev_p[idx_SCALAR] = 0;
#pragma unroll
            for (int q = 0; q < Q; q++)
            {
                dev_p[idx_SCALAR] += dev_f_post[Domain::getIndex<Q>(domain, i, j, k, q)];
            }

#pragma unroll

            for (int q = 0; q < Q; q++)
            {

#ifdef ENABLE_2D
                const CLIP_REAL ga_wa = NSAllen::Equilibrium_new(velSet, q, dev_vel[idx_X], dev_vel[idx_Y]);

#elif defined(ENABLE_3D)
                const CLIP_REAL ga_wa = NSAllen::Equilibrium_new(velSet, q, dev_vel[idx_X], dev_vel[idx_Y], dev_vel[idx_Z]);
#endif

                const CLIP_REAL geq = dev_p[idx_SCALAR] * velSet.wa[q] + ga_wa;
                gneq[q] = dev_f_post[Domain::getIndex<Q>(domain, i, j, k, q)] - geq;
            }

#ifdef ENABLE_2D

            WMRT::convertD2Q9Weighted(gneq, tmp);

            if (dev_c[idx_SCALAR] < 0.0)
                tau = params.tauL;
            else if (dev_c[idx_SCALAR] > 1.0)
                tau = params.tauH;
            else
            {
                tau = params.tauL + dev_c[idx_SCALAR] * (params.tauH - params.tauL);
            }
            const CLIP_REAL s9 = 1.0 / (tau + 0.50);

            tmp[7] = tmp[7] * s9;
            tmp[8] = tmp[8] * s9;

            WMRT::reconvertD2Q9Weighted(tmp, gneq);

            NSAllen::calculateVF<Q, DIM>(velSet, params, gneq, fv, tau, dev_dc[idx_X], dev_dc[idx_Y]);
#elif defined(ENABLE_3D)
            WMRT::convertD3Q19Weighted(gneq, tmp);

            if (dev_c[idx_SCALAR] < 0.0)
                tau = params.tauL;
            else if (dev_c[idx_SCALAR] > 1.0)
                tau = params.tauH;
            else
            {
                tau = params.tauL + dev_c[idx_SCALAR] * (params.tauH - params.tauL);
            }

            const CLIP_REAL s9 = 1.0 / (tau + 0.50);

            tmp[4] = tmp[4] * s9;
            tmp[5] = tmp[5] * s9;
            tmp[6] = tmp[6] * s9;
            tmp[7] = tmp[7] * s9;
            tmp[8] = tmp[8] * s9;

            WMRT::reconvertD3Q19Weighted(tmp, gneq);

            NSAllen::calculateVF<Q, DIM>(velSet, params, gneq, fv, tau, dev_dc[idx_X], dev_dc[idx_Y], dev_dc[idx_Z]);

#endif

            for (CLIP_UINT dim = 0; dim < DIM; dim++)
            {
                if (params.caseType == InputData::CaseType::RTI)
                {
                    fg[dim] = (dev_rho[idx_SCALAR]) * params.gravity[dim];
                }
                else if (params.caseType == InputData::CaseType::Bubble)
                {
                    fg[dim] = (dev_rho[idx_SCALAR] - params.RhoH) * params.gravity[dim];
                }
                else if (params.caseType == InputData::CaseType::Drop)
                {
                    fg[dim] = (dev_rho[idx_SCALAR] - params.RhoL) * params.gravity[dim];
                }
            }

            const CLIP_REAL Fpx = -dev_p[idx_SCALAR] * drho3 * dev_dc[idx_X];
            const CLIP_REAL Fpy = -dev_p[idx_SCALAR] * drho3 * dev_dc[idx_Y];

            const CLIP_REAL Fx = dev_mu[idx_SCALAR] * dev_dc[idx_X] + Fpx + fv[IDX_X] + fg[IDX_X];
            const CLIP_REAL Fy = dev_mu[idx_SCALAR] * dev_dc[idx_Y] + Fpy + fv[IDX_Y] + fg[IDX_Y];

#ifdef ENABLE_3D
            const CLIP_REAL Fpz = -dev_p[idx_SCALAR] * drho3 * dev_dc[idx_Z];
            const CLIP_REAL Fz = dev_mu[idx_SCALAR] * dev_dc[idx_Z] + Fpz + fv[IDX_Z] + fg[IDX_Z];
#endif



            dev_vel[idx_X] = 0;
            dev_vel[idx_Y] = 0;
#ifdef ENABLE_3D
            dev_vel[idx_Z] = 0;
#endif

#pragma unroll
            for (int q = 0; q < Q; q++)
            {
                dev_vel[idx_X] += velSet.ex[q] * dev_f_post[Domain::getIndex<Q>(domain, i, j, k, q)];
                dev_vel[idx_Y] += velSet.ey[q] * dev_f_post[Domain::getIndex<Q>(domain, i, j, k, q)];
#ifdef ENABLE_3D
                dev_vel[idx_Z] += velSet.ez[q] * dev_f_post[Domain::getIndex<Q>(domain, i, j, k, q)];
#endif
            }

            dev_vel[idx_X] = dev_vel[idx_X] + 0.50 * Fx / dev_rho[idx_SCALAR];
            dev_vel[idx_Y] = dev_vel[idx_Y] + 0.50 * Fy / dev_rho[idx_SCALAR];
#ifdef ENABLE_3D
            dev_vel[idx_Z] = dev_vel[idx_Z] + 0.50 * Fz / dev_rho[idx_SCALAR];
#endif
        }
    }

    /**
     * @brief Computes phase field `c` and density `rho` from `g_post`.
     */
    __global__ void kernelMacroscopicg(const WMRT::WMRTvelSet velSet, const InputData::SimParams params, const Domain::DomainInfo domain,
                                       CLIP_REAL *dev_g_post, CLIP_REAL *dev_rho, CLIP_REAL *dev_c)
    {

        constexpr CLIP_UINT Q = WMRT::WMRTvelSet::Q;
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = Domain::getIndex(domain, i, j, k);

        if (Domain::isInside<DIM, true>(domain, i, j, k))
        {

            dev_c[idx_SCALAR] = 0;
#pragma unroll
            for (CLIP_UINT q = 0; q < Q; q++)
            {
                dev_c[idx_SCALAR] += dev_g_post[Domain::getIndex<Q>(domain, i, j, k, q)];
                // dev_c[idx_SCALAR] = 6;
            }

            dev_rho[idx_SCALAR] = (params.RhoL + (dev_c[idx_SCALAR] * (params.RhoH - params.RhoL)));
        }
    }

    /**
     * @brief Applies MRT collision operator for phase field distribution `g`.
     */
    __global__ void kernelCollisionMRTg(const WMRT::WMRTvelSet velSet, const InputData::SimParams params, const Domain::DomainInfo domain,
                                        CLIP_REAL *dev_g, CLIP_REAL *dev_g_post, CLIP_REAL *dev_c, CLIP_REAL *dev_rho, CLIP_REAL *dev_vel, CLIP_REAL *dev_normal)
    {

        const CLIP_REAL wc = 1.0 / (0.50 + 3.0 * params.mobility);
        constexpr CLIP_UINT Q = WMRT::WMRTvelSet::Q;
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = Domain::getIndex(domain, i, j, k);
        const CLIP_UINT idx_X = Domain::getIndex<DIM>(domain, i, j, k, IDX_X);
        const CLIP_UINT idx_Y = Domain::getIndex<DIM>(domain, i, j, k, IDX_Y);

#ifdef ENABLE_3D
        const CLIP_UINT idx_Z = Domain::getIndex<DIM>(domain, i, j, k, IDX_Z);
#endif

        if (Domain::isInside<DIM, true>(domain, i, j, k))
        {

#pragma unroll
            for (CLIP_UINT q = 0; q < Q; q++)
            {

#ifdef ENABLE_2D
                const CLIP_REAL ga_wa = NSAllen::Equilibrium_new(velSet, q, dev_vel[idx_X], dev_vel[idx_Y]);
                const CLIP_REAL e_normal = (velSet.ex[q] * dev_normal[idx_X] + velSet.ey[q] * dev_normal[idx_Y]);
#elif defined(ENABLE_3D)
                const CLIP_REAL ga_wa = NSAllen::Equilibrium_new(velSet, q, dev_vel[idx_X], dev_vel[idx_Y], dev_vel[idx_Z]);
                const CLIP_REAL e_normal = (velSet.ex[q] * dev_normal[idx_X] + velSet.ey[q] * dev_normal[idx_Y] + velSet.ez[q] * dev_normal[idx_Z]);
#endif

                const CLIP_REAL eF = ((3.0 * (params.mobility) * (1.0 - 4.0 * ((dev_c[idx_SCALAR] - 0.50) * (dev_c[idx_SCALAR] - 0.50)))) / (params.interfaceWidth)) * e_normal;
                const CLIP_REAL hlp = velSet.wa[q] * eF;
                const CLIP_REAL heq = dev_c[idx_SCALAR] * (ga_wa + velSet.wa[q]) + hlp;
                dev_g[Domain::getIndex<Q>(domain, i, j, k, q)] = dev_g_post[Domain::getIndex<Q>(domain, i, j, k, q)] * (1.0 - (wc)) + heq * (wc);
            }
        }
    }

    /**
     * @brief Applies MRT collision operator for momentum distribution `f`, including force correction.
     */
    __global__ void kernelCollisionMRTf(const WMRT::WMRTvelSet velSet, const InputData::SimParams params, const Domain::DomainInfo domain,
                                        CLIP_REAL *dev_f, CLIP_REAL *dev_f_post, CLIP_REAL *dev_p, CLIP_REAL *dev_c, CLIP_REAL *dev_dc,
                                        CLIP_REAL *dev_mu, CLIP_REAL *dev_rho, CLIP_REAL *dev_vel, CLIP_REAL *dev_normal)
    {

        constexpr CLIP_UINT Q = WMRT::WMRTvelSet::Q;
        CLIP_REAL gneq[Q], tmp[Q], ga_wa[Q], hlp[Q], fv[DIM], fg[DIM], tau, s9;
        const CLIP_REAL drho3 = (params.RhoH - params.RhoL) / 3.0;
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = Domain::getIndex(domain, i, j, k);
        const CLIP_UINT idx_X = Domain::getIndex<DIM>(domain, i, j, k, IDX_X);
        const CLIP_UINT idx_Y = Domain::getIndex<DIM>(domain, i, j, k, IDX_Y);

#ifdef ENABLE_3D
        const CLIP_UINT idx_Z = Domain::getIndex<DIM>(domain, i, j, k, IDX_Z);
#endif

        if (Domain::isInside<DIM, true>(domain, i, j, k))
        {

#pragma unroll

            for (CLIP_UINT q = 0; q < Q; q++)
            {
#ifdef ENABLE_2D
                ga_wa[q] = NSAllen::Equilibrium_new(velSet, q, dev_vel[idx_X], dev_vel[idx_Y]);

#elif defined(ENABLE_3D)
                ga_wa[q] = NSAllen::Equilibrium_new(velSet, q, dev_vel[idx_X], dev_vel[idx_Y], dev_vel[idx_Z]);
#endif
                const CLIP_REAL geq = dev_p[idx_SCALAR] * velSet.wa[q] + ga_wa[q];
                gneq[q] = dev_f_post[Domain::getIndex<Q>(domain, i, j, k, q)] - geq;
            }

#ifdef ENABLE_2D

            WMRT::convertD2Q9Weighted(gneq, tmp);

            if (dev_c[idx_SCALAR] < 0.0)
                tau = params.tauL;
            else if (dev_c[idx_SCALAR] > 1.0)
                tau = params.tauH;
            else
            {
                tau = params.tauL + dev_c[idx_SCALAR] * (params.tauH - params.tauL);
            }
            s9 = 1.0 / (tau + 0.50);

            tmp[7] = tmp[7] * s9;
            tmp[8] = tmp[8] * s9;

            WMRT::reconvertD2Q9Weighted(tmp, gneq);

            NSAllen::calculateVF<Q, DIM>(velSet, params, gneq, fv, tau, dev_dc[idx_X], dev_dc[idx_Y]);
#elif defined(ENABLE_3D)
            WMRT::convertD3Q19Weighted(gneq, tmp);

            if (dev_c[idx_SCALAR] < 0.0)
                tau = params.tauL;
            else if (dev_c[idx_SCALAR] > 1.0)
                tau = params.tauH;
            else
            {
                tau = params.tauL + dev_c[idx_SCALAR] * (params.tauH - params.tauL);
            }

            s9 = 1.0 / (tau + 0.50);

            tmp[4] = tmp[4] * s9;
            tmp[5] = tmp[5] * s9;
            tmp[6] = tmp[6] * s9;
            tmp[7] = tmp[7] * s9;
            tmp[8] = tmp[8] * s9;

            WMRT::reconvertD3Q19Weighted(tmp, gneq);

            NSAllen::calculateVF<Q, DIM>(velSet, params, gneq, fv, tau, dev_dc[idx_X], dev_dc[idx_Y], dev_dc[idx_Z]);

#endif

            for (CLIP_UINT dim = 0; dim < DIM; dim++)
            {
                if (params.caseType == InputData::CaseType::RTI)
                {
                    fg[dim] = (dev_rho[idx_SCALAR]) * params.gravity[dim];
                }
                else if (params.caseType == InputData::CaseType::Bubble)
                {
                    fg[dim] = (dev_rho[idx_SCALAR] - params.RhoH) * params.gravity[dim];
                }
                else if (params.caseType == InputData::CaseType::Drop)
                {
                    fg[dim] = (dev_rho[idx_SCALAR] - params.RhoL) * params.gravity[dim];
                }
            }
            
            const CLIP_REAL Fpx = -dev_p[idx_SCALAR] * drho3 * dev_dc[idx_X];
            const CLIP_REAL Fpy = -dev_p[idx_SCALAR] * drho3 * dev_dc[idx_Y];

            const CLIP_REAL Fx = dev_mu[idx_SCALAR] * dev_dc[idx_X] + Fpx + fv[IDX_X] + fg[IDX_X];
            const CLIP_REAL Fy = dev_mu[idx_SCALAR] * dev_dc[idx_Y] + Fpy + fv[IDX_Y] + fg[IDX_Y];

#ifdef ENABLE_3D
            const CLIP_REAL Fpz = -dev_p[idx_SCALAR] * drho3 * dev_dc[idx_Z];
            const CLIP_REAL Fz = dev_mu[idx_SCALAR] * dev_dc[idx_Z] + Fpz + fv[IDX_Z] + fg[IDX_Z];
#endif

#pragma unroll
            for (CLIP_UINT q = 0; q < Q; q++)
            {
#ifdef ENABLE_2D
                const CLIP_REAL eF = velSet.ex[q] * Fx + velSet.ey[q] * Fy;

#elif defined(ENABLE_3D)
                const CLIP_REAL eF = velSet.ex[q] * Fx + velSet.ey[q] * Fy + velSet.ez[q] * Fz;
#endif
                hlp[q] = 3.0 * velSet.wa[q] * eF / dev_rho[idx_SCALAR];
                const CLIP_REAL feq = dev_p[idx_SCALAR] * velSet.wa[q] + ga_wa[q] - 0.50 * hlp[q];
                gneq[q] = dev_f_post[Domain::getIndex<Q>(domain, i, j, k, q)] - feq;
            }

#ifdef ENABLE_2D

            WMRT::convertD2Q9Weighted(gneq, tmp);

            if (dev_c[idx_SCALAR] < 0.0)
                tau = params.tauL;
            else if (dev_c[idx_SCALAR] > 1.0)
                tau = params.tauH;
            else
            {
                tau = params.tauL + dev_c[idx_SCALAR] * (params.tauH - params.tauL);
            }

            s9 = 1.0 / (tau + 0.50);

            tmp[7] = tmp[7] * s9;
            tmp[8] = tmp[8] * s9;

            WMRT::reconvertD2Q9Weighted(tmp, gneq);

#elif defined(ENABLE_3D)
            WMRT::convertD3Q19Weighted(gneq, tmp);

            if (dev_c[idx_SCALAR] < 0.0)
                tau = params.tauL;
            else if (dev_c[idx_SCALAR] > 1.0)
                tau = params.tauH;
            else
            {
                tau = params.tauL + dev_c[idx_SCALAR] * (params.tauH - params.tauL);
            }

            s9 = 1.0 / (tau + 0.50);

            tmp[4] = tmp[4] * s9;
            tmp[5] = tmp[5] * s9;
            tmp[6] = tmp[6] * s9;
            tmp[7] = tmp[7] * s9;
            tmp[8] = tmp[8] * s9;

            WMRT::reconvertD3Q19Weighted(tmp, gneq);

#endif

#pragma unroll
            for (int q = 0; q < Q; q++)
            {
                dev_f[Domain::getIndex<Q>(domain, i, j, k, q)] = dev_f_post[Domain::getIndex<Q>(domain, i, j, k, q)] - gneq[q] + hlp[q];
            }
        }
    }

    /**
     * @brief Initializes host-side phase field variable `c` based on geometry and case type.
     */
    void NSAllen::initialCondition()
    {
        using namespace clip;

        for (CLIP_UINT i = m_info.ghostDomainMinIdx[IDX_X]; i <= m_info.ghostDomainMaxIdx[IDX_X]; i++)
        {
            for (CLIP_UINT j = m_info.ghostDomainMinIdx[IDX_Y]; j <= m_info.ghostDomainMaxIdx[IDX_Y]; j++)
            {
                for (CLIP_UINT k = m_info.ghostDomainMinIdx[IDX_Z]; k <= m_info.ghostDomainMaxIdx[IDX_Z]; k++)
                {
                    const CLIP_UINT idx_SCALAR = Domain::getIndex(m_info, i, j, k);

                    // Physical coordinate relative to center
                    const CLIP_REAL x = static_cast<CLIP_REAL>(i);
                    const CLIP_REAL y = static_cast<CLIP_REAL>(j);
                    const CLIP_REAL z = (DIM == 3) ? static_cast<CLIP_REAL>(k) : 0.0;

                    CLIP_REAL sdf_val = 1e10;

                    // Select based on case
                    switch (m_params.caseType)
                    {
                    case InputData::CaseType::Bubble:
                    {
                        sdf_val = Geometry::sdf(m_geomPool, 0, x, y, z);
                        m_DA->hostDA.host_c[idx_SCALAR] = 0.50 + 0.50 * tanh(2.0 * sdf_val / m_params.interfaceWidth);
                        break;
                    }

                    case InputData::CaseType::Drop:
                    {
                        sdf_val = Geometry::sdf(m_geomPool, 0, x, y, z);
                        m_DA->hostDA.host_c[idx_SCALAR] = 0.50 - 0.50 * tanh(2.0 * sdf_val / m_params.interfaceWidth);
                        break;
                    }
                    case InputData::CaseType::Jet:
                    {
                        sdf_val = Geometry::sdf(m_geomPool, 0, x, y, z);
                        m_DA->hostDA.host_c[idx_SCALAR] = 0.50 - 0.50 * tanh(2.0 * sdf_val / m_params.interfaceWidth);
                        break;
                    }
                    case InputData::CaseType::RTI:
                    {
                        // Assume ID=0 corresponds to perturbation field
                        sdf_val = Geometry::sdf(m_geomPool, 0, x, y, z);
                        // printf("sdf: %f \n", sdf_val);
                        m_DA->hostDA.host_c[idx_SCALAR] = 0.5 + 0.5 * tanh(2.0 * sdf_val / m_params.interfaceWidth);
                        break;
                    }
                    default:
                        m_DA->hostDA.host_c[idx_SCALAR] = 0.0;
                        break;
                    }
                }
            }
        }
    }

    /**
     * @brief Performs MRT collision for both momentum and phase fields.
     * Applies Allen–Cahn interface tracking and force coupling.
     */
    void NSAllen::collision()
    {

        normal_FD<<<dimGrid, dimBlock>>>(m_info, m_DA->deviceDA.dev_dc, m_DA->deviceDA.dev_normal);

        kernelCollisionMRTg<<<dimGrid, dimBlock>>>(m_velset, m_params, m_info,
                                                   m_DA->deviceDA.dev_g, m_DA->deviceDA.dev_g_post, m_DA->deviceDA.dev_c, m_DA->deviceDA.dev_rho, m_DA->deviceDA.dev_vel, m_DA->deviceDA.dev_normal);

        kernelCollisionMRTf<<<dimGrid, dimBlock>>>(m_velset, m_params, m_info,
                                                   m_DA->deviceDA.dev_f, m_DA->deviceDA.dev_f_post, m_DA->deviceDA.dev_p, m_DA->deviceDA.dev_c, m_DA->deviceDA.dev_dc, m_DA->deviceDA.dev_mu, m_DA->deviceDA.dev_rho, m_DA->deviceDA.dev_vel, m_DA->deviceDA.dev_normal);

        cudaDeviceSynchronize();
    }

    /**
     * @brief Performs the streaming step for both `f` and `g` distributions using lattice velocities.
     */
    void NSAllen::streaming()
    {
        constexpr CLIP_UINT Q = WMRT::WMRTvelSet::Q;
        if (m_boundary->isFreeConvect)
        {
            m_DA->copyDevice(m_DA->deviceDA.dev_g_prev, m_DA->deviceDA.dev_g_post, "dev_g_prev", Q);
            m_DA->copyDevice(m_DA->deviceDA.dev_f_prev, m_DA->deviceDA.dev_f_post, "dev_f_prev", Q);
        }

        kernelStreaming<<<dimGrid, dimBlock>>>(m_velset, m_info, m_DA->deviceDA.dev_f, m_DA->deviceDA.dev_f_post);
        kernelStreaming<<<dimGrid, dimBlock>>>(m_velset, m_info, m_DA->deviceDA.dev_g, m_DA->deviceDA.dev_g_post);

        cudaDeviceSynchronize();
    }

    /**
     * @brief Updates macroscopic fields: density, pressure, velocity, chemical potential, and gradients.
     */
    void NSAllen::macroscopic()
    {

        kernelMacroscopicg<<<dimGrid, dimBlock>>>(m_velset, m_params, m_info,
                                                  m_DA->deviceDA.dev_g_post, m_DA->deviceDA.dev_rho, m_DA->deviceDA.dev_c);

        Chemical_Potential<<<dimGrid, dimBlock>>>(m_params, m_info, m_DA->deviceDA.dev_c, m_DA->deviceDA.dev_mu);
        Isotropic_Gradient<<<dimGrid, dimBlock>>>(m_info, m_DA->deviceDA.dev_c, m_DA->deviceDA.dev_dc);
        kernelMacroscopicf<<<dimGrid, dimBlock>>>(m_velset, m_params, m_info,
                                                  m_DA->deviceDA.dev_p, m_DA->deviceDA.dev_rho, m_DA->deviceDA.dev_c, m_DA->deviceDA.dev_f_post, m_DA->deviceDA.dev_dc, m_DA->deviceDA.dev_vel, m_DA->deviceDA.dev_mu);
        cudaDeviceSynchronize();
    }

    /**
     * @brief Runs one full LBM timestep: collision, boundary conditions, streaming, and macroscopic update.
     */
    void NSAllen::solve()
    {
        collision();
        applyPeriodicBoundary();
        velocityBoundary(m_DA->deviceDA.dev_c, m_DA->deviceDA.dev_f, m_DA->deviceDA.dev_g);
        streaming();
        applyWallBoundary();
        applyFreeConvectBoundary();
        applyNeumannBoundary();
        macroscopic();
        cudaDeviceSynchronize();
    }

    /**
     * @brief Initializes device-side fields including `rho`, `p`, `c`, `f`, `g`, and gradients.
     */
    void NSAllen::deviceInitializer()
    {

        Chemical_Potential<<<dimGrid, dimBlock>>>(m_params, m_info, m_DA->deviceDA.dev_c, m_DA->deviceDA.dev_mu);

        Isotropic_Gradient<<<dimGrid, dimBlock>>>(m_info, m_DA->deviceDA.dev_c, m_DA->deviceDA.dev_dc);

        normal_FD<<<dimGrid, dimBlock>>>(m_info, m_DA->deviceDA.dev_dc, m_DA->deviceDA.dev_normal);

        KernelInitializeDistributions<<<dimGrid, dimBlock>>>(m_velset, m_params, m_info, m_DA->deviceDA.dev_f, m_DA->deviceDA.dev_g, m_DA->deviceDA.dev_f_post, m_DA->deviceDA.dev_g_post,
                                                             m_DA->deviceDA.dev_c, m_DA->deviceDA.dev_rho, m_DA->deviceDA.dev_p, m_DA->deviceDA.dev_vel, m_DA->deviceDA.dev_normal);
    }

    /**
     * @brief Applies periodic boundary conditions to all fields.
     */
    void NSAllen::applyPeriodicBoundary()
    {

        constexpr CLIP_UINT Q = WMRT::WMRTvelSet::Q;

        periodicBoundary<Q>(m_DA->deviceDA.dev_f, m_DA->deviceDA.dev_g);
        periodicBoundary<SCALAR_FIELD>(m_DA->deviceDA.dev_c);
        cudaDeviceSynchronize();
    }

    /**
     * @brief Applies wall boundary condition using bounce-back and mirror for `c`.
     */
    void NSAllen::applyWallBoundary()
    {
        constexpr CLIP_UINT Q = WMRT::WMRTvelSet::Q;
        constexpr CLIP_UINT dof = WMRT::wallBCMap::Q;
        wallBoundary<Q, dof>(m_DA->deviceDA.dev_f, m_DA->deviceDA.dev_f_post, m_DA->deviceDA.dev_g, m_DA->deviceDA.dev_g_post);
        mirrorBoundary(m_DA->deviceDA.dev_c);
        cudaDeviceSynchronize();
    }

    /**
     * @brief Applies free-convective outlet boundary condition using extrapolation and mirroring.
     */
    void NSAllen::applyFreeConvectBoundary()
    {
        constexpr CLIP_UINT Q = WMRT::WMRTvelSet::Q;
        constexpr CLIP_UINT dof = WMRT::wallBCMap::Q;
        freeConvectBoundary<Q, dof>(m_DA->deviceDA.dev_vel, m_DA->deviceDA.dev_f_post, m_DA->deviceDA.dev_f_prev, m_DA->deviceDA.dev_g_post, m_DA->deviceDA.dev_g_prev);
        mirrorBoundary(m_DA->deviceDA.dev_c);
        cudaDeviceSynchronize();
    }

    /**
     * @brief Applies Neumann (zero-gradient) boundary condition to distributions and phase field.
     */
    void NSAllen::applyNeumannBoundary()
    {
        constexpr CLIP_UINT Q = WMRT::WMRTvelSet::Q;
        constexpr CLIP_UINT dof = WMRT::wallBCMap::Q;
        NeumannBoundary<Q, dof>(m_DA->deviceDA.dev_f_post, m_DA->deviceDA.dev_g_post);
        mirrorBoundary(m_DA->deviceDA.dev_c);
        cudaDeviceSynchronize();
    }
}