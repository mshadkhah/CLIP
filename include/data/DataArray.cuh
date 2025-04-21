#pragma once
#include "includes.h"
#include "InputData.cuh"
#include "Domain.cuh"
#include "WMRT.cuh"



const CLIP_REAL pi = 3.14159265358979L;

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

#define THREAD_IDX_X (threadIdx.x + blockIdx.x * blockDim.x)
#define THREAD_IDX_Y (threadIdx.y + blockIdx.y * blockDim.y)
#define THREAD_IDX_Z (threadIdx.z + blockIdx.z * blockDim.z)

namespace clip
{

    class DataArray
    {

    public:
        explicit DataArray(const InputData& idata, const Domain& domain);

        template <typename T>
        void allocateOnDevice(T *&devPtr, const char *name, CLIP_UINT ndof = SCALAR_FIELD)
        {
            cudaMalloc((void **)&devPtr, ndof * m_domain->domainSize * sizeof(T));
            cudaCheckErrors(("cudaMalloc '" + std::string(name) + "' fail").c_str());
            cudaDeviceSynchronize();
        }

        template <typename T>
        void allocateOnHost(T *&hostPtr, const char *name, CLIP_UINT ndof = SCALAR_FIELD)
        {
            // Standard malloc or aligned malloc if needed
            hostPtr = (T *)malloc(ndof * m_domain->domainSize * sizeof(T));
            if (!hostPtr)
            {
                std::cerr << "Host malloc failed for: " << name << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        template <typename T>
        void copyToDevice(T*& devPtr, const T* hostPtr, const char* name, CLIP_UINT ndof = SCALAR_FIELD)
        {
            cudaMemcpy(devPtr, hostPtr, ndof * m_domain->domainSize * sizeof(T), cudaMemcpyHostToDevice);
            cudaCheckErrors(("copyToDevice failed for " + std::string(name)).c_str());
        }
        

        template <typename T>
        void copyFromDevice(T*&hostPtr, const T *devPtr, const char *name, CLIP_UINT ndof = SCALAR_FIELD)
        {
            cudaMemcpy(hostPtr, devPtr, ndof * m_domain->domainSize * sizeof(T), cudaMemcpyDeviceToHost);
            cudaCheckErrors(("copyFromDevice failed for " + std::string(name)).c_str());
        }

        template <typename T, size_t N>
        void symbolOnDevice(const T (&symbol)[N], const T *hostPtr, const char *name)
        {
            cudaMemcpyToSymbol(symbol, hostPtr, sizeof(T) * N);
            cudaCheckErrors((std::string("cudaMemcpyToSymbol '") + name + "' failed").c_str());
            cudaDeviceSynchronize();
        }

        template <typename T>
        void symbolOnDevice(const T &symbol, const T *hostPtr, const char *name)
        {
            cudaMemcpyToSymbol(symbol, hostPtr, sizeof(T));
            cudaCheckErrors((std::string("cudaMemcpyToSymbol '") + name + "' failed").c_str());
            cudaDeviceSynchronize();
        }

        dim3 dimBlock, dimGrid;


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
        };

        struct hostDataArray
        {
            CLIP_REAL *host_c;
            CLIP_REAL *host_rho;
            CLIP_REAL *host_p;
            CLIP_REAL *host_vel;
        };



        deviceDataArray deviceDA;
        hostDataArray hostDA;




    void createVectors();
    void updateDevice();
    void updateHost();


    private:

        const InputData* m_idata;
        const Domain* m_domain;

    };

}
