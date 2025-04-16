#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <DataTypes.cuh>
#include <Domain.cuh>

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
        explicit DataArray(InputData idata);

     

        template <typename T>
        void allocateOnDevice(T *&devPtr, const char *name, CLIP_UINT ndof = SCALAR)
        {
            cudaMalloc((void **)&devPtr, ndof * m_domain.domainSize * sizeof(T));
            cudaCheckErrors(("cudaMalloc '" + std::string(name) + "' fail").c_str());
            cudaDeviceSynchronize();
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
        
    private:
        CLIP_UINT m_nVelocity;
        CLIP_UINT latticeSize;
        CLIP_UINT domainSize;
        CLIP_UINT *m_domainExtent;
        CLIP_UINT *m_domainExtentGhosted;

        InputData m_idata;
        Domain m_domain;

    protected:



    };

}
