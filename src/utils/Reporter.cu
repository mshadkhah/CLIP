// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file Reporter.cu
 * @brief Reporting and diagnostics for CLIP simulations.
 *
 * @details
 * This file implements the `Reporter` class used to log and track simulation progress.
 * It includes device and host-side code for computing global quantities (e.g., total concentration).
 * The class prints diagnostics like time step, domain size, and integral of scalar fields
 * at user-defined intervals. It also detects divergence (e.g., NaNs in concentration).
 *
 * CUDA kernels are used for fast parallel reductions on the device.
 *
 * @author
 * Mehdi Shadkhah
 *
 * @date
 * 2025
 */

#include "Reporter.cuh"

namespace clip
{

    /**
     * @brief Constructs a Reporter object.
     * @param DA Reference to the simulation data array.
     * @param idata Reference to input configuration.
     * @param domain Reference to domain information.
     * @param ti Time tracking info for simulation steps and wall time.
     */

    Reporter::Reporter(DataArray &DA, const InputData &idata, const Domain &domain, const TimeInfo &ti)
        : m_DA(&DA), m_idata(&idata), m_domain(&domain), m_ti(&ti)
    {
    }

    /**
     * @brief CUDA kernel for parallel sum reduction over a scalar field.
     *
     * @param data Pointer to device array containing values to be summed.
     * @param partialSums Output array for storing per-block partial results.
     * @param domain Domain info including ghost zone and extents.
     */

    __global__ void sumReductionKernel(const CLIP_REAL *__restrict__ data, CLIP_REAL *partialSums,
                                       Domain::DomainInfo domain)
    {
        extern __shared__ double sdata[];
        int tid = threadIdx.x;

        int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

        int Nx = domain.extent[IDX_X];
        int Ny = domain.extent[IDX_Y];
        int Nz = domain.extent[IDX_Z];

        int i = globalIdx % Nx;
        int j = (globalIdx / Nx) % Ny;
        int k = globalIdx / (Nx * Ny);

        double value = 0.0;
        if (Domain::isInside<DIM, true>(domain, i, j, k))
        {
            int idx = Domain::getIndex(domain, i, j, k);
            value = data[idx];
        }

        sdata[tid] = value;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }

        if (tid == 0)
            partialSums[blockIdx.x] = sdata[0];
    }

    /**
     * @brief Computes the total sum of a scalar field on the device.
     *
     * @param dev_data Device pointer to the scalar field (e.g., concentration).
     * @param dof Number of degrees of freedom (number of elements to sum).
     * @return Total sum as a double.
     */

    CLIP_REAL Reporter::sum(CLIP_REAL *dev_data, CLIP_UINT dof)
    {
        const int threadsPerBlock = 256;
        const int blocks = (dof + threadsPerBlock - 1) / threadsPerBlock;

        double *dev_partialSums;
        cudaMalloc(&dev_partialSums, blocks * sizeof(double));

        sumReductionKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(CLIP_REAL)>>>(dev_data, dev_partialSums, m_domain->info);

        std::vector<double> h_partialSums(blocks);
        cudaMemcpy(h_partialSums.data(), dev_partialSums, blocks * sizeof(CLIP_REAL), cudaMemcpyDeviceToHost);
        cudaFree(dev_partialSums);

        double totalSum = 0.0;
        for (int i = 0; i < blocks; ++i)
            totalSum += h_partialSums[i];

        return totalSum;
    }

    /**
     * @brief Prints simulation progress and scalar diagnostics at regular intervals.
     *
     * @details
     * This includes the step number, sum of the concentration field, and domain size.
     * If NaNs are detected in the scalar field, it logs an error indicating divergence.
     */

    void Reporter::print()
    {
        const CLIP_UINT step = m_ti->getCurrentStep();

        if (step % m_idata->params.reportInterval == 0)
        {
            CLIP_REAL sumC = sum(m_DA->deviceDA.dev_c, m_domain->domainSize);

            std::ostringstream oss;
            oss << "\n===================== Solver Status =====================\n"
                << "Step        : " << step << "\n"
                // << "Time        : " << time << "\n"
                << "Sum(c)      : " << sumC << "\n"
                << "Domain Size : " << m_domain->domainSize << "\n"
                << "---------------------------------------------------------";

            Logger::Info(oss.str());
            if (std::isnan(sumC))
                Logger::Error("Solver has diverged.");
        }
    }

}