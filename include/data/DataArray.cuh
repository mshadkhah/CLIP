// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file
 * @brief Defines the DataArray class for managing all host/device memory allocation,
 *        data transfer, and file I/O in the CLIP LBM framework.
 *
 * The DataArray class provides:
 * - Allocation of memory on both CPU and GPU
 * - Utilities for copying data between host and device
 * - Management of simulation fields like velocity, pressure, and distribution functions
 * - Support for symbol memory transfer for constants
 * - File I/O for saving and loading raw binary field data
 *
 * This class is central to memory layout and performance in the CUDA-accelerated
 * CLIP simulation backend for multiphase/interfacial flow using the Lattice Boltzmann Method.
 */


#pragma once
#include "includes.h"
#include "InputData.cuh"
#include "Domain.cuh"
#include "WMRT.cuh"
#include "Boundary.cuh"

/// CUDA error checking macro.
#define cudaCheckErrors(msg)                                   \
    do                                                         \
    {                                                          \
        cudaError_t __err = cudaGetLastError();                \
        if (__err != cudaSuccess)                              \
        {                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err),            \
                    __FILE__, __LINE__);                       \
            fprintf(stderr, "*** FAILED - ABORTING\n");        \
            exit(1);                                           \
        }                                                      \
    } while (0)

/// helper macro to free up the memory.
#define SAFE_CUDA_FREE(ptr) if (ptr) { cudaFree(ptr); ptr = nullptr; }
#define SAFE_FREE(ptr) if (ptr) { free(ptr); ptr = nullptr; }

/// 1D thread indexing macros
#define THREAD_IDX_X (threadIdx.x + blockIdx.x * blockDim.x)
#define THREAD_IDX_Y (threadIdx.y + blockIdx.y * blockDim.y)
#define THREAD_IDX_Z (threadIdx.z + blockIdx.z * blockDim.z)

namespace clip
{

    /**
     * @brief Manages all host and device memory allocations for the CLIP LBM solver.
     *
     * This class handles allocation, transfer, and file I/O for all primary simulation fields
     * used in the Lattice Boltzmann Method (LBM), including velocity, pressure, and distribution functions.
     */
    class DataArray
    {

    public:
        /**
         * @brief Construct a DataArray manager with simulation inputs and domain info.
         * @param idata Input parameters
         * @param domain Simulation domain object
         * @param boundary Boundary condition object
         */
        explicit DataArray(const InputData &idata, const Domain &domain, const Boundary &boundary);


         /// Destructor
        ~DataArray();

        /**
         * @brief Allocates device memory on GPU for a given pointer.
         * @tparam T Data type
         * @param[out] devPtr Pointer to be allocated on device
         * @param name Debugging label
         * @param ndof Number of degrees of freedom (default = scalar)
         */
        template <typename T>
        void allocateOnDevice(T *&devPtr, const std::string &name, CLIP_UINT ndof = SCALAR_FIELD)
        {
            cudaMalloc((void **)&devPtr, ndof * m_domain->domainSize * sizeof(T));
            cudaCheckErrors(("cudaMalloc '" + name + "' fail").c_str());
            cudaDeviceSynchronize();
        }

        /**
         * @brief Allocates host memory (CPU) for a given pointer.
         * @tparam T Data type
         * @param[out] hostPtr Pointer to be allocated on host
         * @param name Debugging label
         * @param ndof Number of degrees of freedom
         */
        template <typename T>
        void allocateOnHost(T *&hostPtr, const std::string &name, CLIP_UINT ndof = SCALAR_FIELD)
        {
            hostPtr = (T *)malloc(ndof * m_domain->domainSize * sizeof(T));
            if (!hostPtr)
            {
                std::cerr << "Host malloc failed for: " << name << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        /**
         * @brief Copies data from host to device.
         * @tparam T Data type
         * @param[out] devPtr Device pointer destination
         * @param hostPtr Host source pointer
         * @param name Debug label
         * @param ndof Number of degrees of freedom
         */
        template <typename T>
        void copyToDevice(T *&devPtr, const T *hostPtr, const std::string &name, CLIP_UINT ndof = SCALAR_FIELD)
        {
            cudaMemcpy(devPtr, hostPtr, ndof * m_domain->domainSize * sizeof(T), cudaMemcpyHostToDevice);
            cudaCheckErrors(("copyToDevice failed for " + name).c_str());
        }

        /**
         * @brief Copies data from device to host.
         * @tparam T Data type
         * @param[out] hostPtr Host pointer destination
         * @param devPtr Device source pointer
         * @param name Debug label
         * @param ndof Number of degrees of freedom
         */
        template <typename T>
        void copyFromDevice(T *&hostPtr, const T *devPtr, const std::string &name, CLIP_UINT ndof = SCALAR_FIELD)
        {
            cudaMemcpy(hostPtr, devPtr, ndof * m_domain->domainSize * sizeof(T), cudaMemcpyDeviceToHost);
            cudaCheckErrors(("copyFromDevice failed for " + name).c_str());
        }

        /**
         * @brief Copies data between device pointers.
         * @tparam T Data type
         * @param[out] destPtr Destination device pointer
         * @param sourcePtr Source device pointer
         * @param name Debug label
         * @param ndof Number of degrees of freedom
         */
        template <typename T>
        void copyDevice(T *&destPtr, const T *sourcePtr, const std::string &name, CLIP_UINT ndof = SCALAR_FIELD)
        {
            cudaMemcpy(destPtr, sourcePtr, ndof * m_domain->domainSize * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaCheckErrors(("copyFromDeviceToDevice failed for " + name).c_str());
        }

        /**
         * @brief Copies a constant array to device symbol memory.
         * @tparam T Data type
         * @tparam N Array size
         * @param symbol Symbol destination
         * @param hostPtr Source data
         * @param name Debug label
         */
        template <typename T, size_t N>
        void symbolOnDevice(const T (&symbol)[N], const T *hostPtr, const std::string &name)
        {
            cudaMemcpyToSymbol(symbol, hostPtr, sizeof(T) * N);
            cudaCheckErrors(("cudaMemcpyToSymbol '" + name + "' failed").c_str());
            cudaDeviceSynchronize();
        }

        /**
         * @brief Copies a single object to device symbol memory.
         * @tparam T Data type
         * @param symbol Symbol reference
         * @param hostPtr Source data
         * @param name Debug label
         */
        template <typename T>
        void symbolOnDevice(const T &symbol, const T *hostPtr, const std::string &name)
        {
            cudaMemcpyToSymbol(symbol, hostPtr, sizeof(T));
            cudaCheckErrors(("cudaMemcpyToSymbol '" + name + "' failed").c_str());
            cudaDeviceSynchronize();
        }

        /**
         * @brief Writes a host-side array to a binary file.
         * @tparam T Data type
         * @param hostPtr Pointer to data
         * @param filename Output file path
         * @param size Number of elements
         */
        template <typename T>
        void writeHostToFile(const T *hostPtr, const std::string &filename, CLIP_UINT size)
        {
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open())
            {
                Logger::Error("Failed to open file for saving: " + filename);
                return;
            }
            file.write(reinterpret_cast<const char *>(hostPtr), size * sizeof(T));
            file.close();
        }

        /**
         * @brief Reads a host-side array from a binary file.
         * @tparam T Data type
         * @param[out] hostPtr Pointer to write into
         * @param filename Input file path
         * @param size Number of elements
         */
        template <typename T>
        void readHostFromFile(T *hostPtr, const std::string &filename, CLIP_UINT size)
        {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open())
            {
                Logger::Error("Failed to open file for reading: " + filename);
                return;
            }

            file.read(reinterpret_cast<char *>(hostPtr), size * sizeof(T));
            file.close();
        }

        /// CUDA thread/block configuration for kernels
        dim3 dimBlock, dimGrid;

        /**
         * @brief Struct holding all device-side (GPU) pointers for LBM fields.
         */
        struct deviceDataArray
        {
            CLIP_REAL *dev_f;
            CLIP_REAL *dev_f_post;
            CLIP_REAL *dev_g;
            CLIP_REAL *dev_g_post;
            CLIP_REAL *dev_rho;
            CLIP_REAL *dev_p;
            CLIP_REAL *dev_c;
            CLIP_REAL *dev_dc;
            CLIP_REAL *dev_vel;
            CLIP_REAL *dev_mu;
            CLIP_REAL *dev_normal;

            CLIP_REAL *dev_f_prev;
            CLIP_REAL *dev_g_prev;
        };

        /**
         * @brief Struct holding all host-side (CPU) pointers for LBM fields.
         */
        struct hostDataArray
        {
            CLIP_REAL *host_c;
            CLIP_REAL *host_rho;
            CLIP_REAL *host_p;
            CLIP_REAL *host_vel;

            CLIP_REAL *host_f;
            CLIP_REAL *host_f_post;
            CLIP_REAL *host_g;
            CLIP_REAL *host_g_post;

            CLIP_REAL *host_f_prev;
            CLIP_REAL *host_g_prev;
        };

        deviceDataArray deviceDA;
        hostDataArray hostDA;

        /**
         * @brief Allocate and initialize all host/device field arrays.
         */
        void createVectors();

        /**
         * @brief Copy all host-side fields to device.
         */
        void updateDevice();

        /**
         * @brief Copy all device-side fields to host.
         */
        void updateHost();

    private:
        const InputData *m_idata;   ///< Simulation input parameters
        const Domain *m_domain;     ///< Mesh and domain info
        const Boundary *m_boundary; ///< Boundary condition interface
    };

}
