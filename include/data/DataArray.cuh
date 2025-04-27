#pragma once
#include "includes.h"
#include "InputData.cuh"
#include "Domain.cuh"
#include "WMRT.cuh"
#include "Boundary.cuh"



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
        explicit DataArray(const InputData& idata, const Domain& domain, const Boundary& boundary);

        template <typename T>
        void allocateOnDevice(T*& devPtr, const std::string& name, CLIP_UINT ndof = SCALAR_FIELD)
        {
            cudaMalloc((void**)&devPtr, ndof * m_domain->domainSize * sizeof(T));
            cudaCheckErrors(("cudaMalloc '" + name + "' fail").c_str());
            cudaDeviceSynchronize();
        }
        
        template <typename T>
        void allocateOnHost(T*& hostPtr, const std::string& name, CLIP_UINT ndof = SCALAR_FIELD)
        {
            hostPtr = (T*)malloc(ndof * m_domain->domainSize * sizeof(T));
            if (!hostPtr)
            {
                std::cerr << "Host malloc failed for: " << name << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        
        template <typename T>
        void copyToDevice(T*& devPtr, const T* hostPtr, const std::string& name, CLIP_UINT ndof = SCALAR_FIELD)
        {
            cudaMemcpy(devPtr, hostPtr, ndof * m_domain->domainSize * sizeof(T), cudaMemcpyHostToDevice);
            cudaCheckErrors(("copyToDevice failed for " + name).c_str());
        }
        
        template <typename T>
        void copyFromDevice(T*& hostPtr, const T* devPtr, const std::string& name, CLIP_UINT ndof = SCALAR_FIELD)
        {
            cudaMemcpy(hostPtr, devPtr, ndof * m_domain->domainSize * sizeof(T), cudaMemcpyDeviceToHost);
            cudaCheckErrors(("copyFromDevice failed for " + name).c_str());
        }
        
        template <typename T>
        void copyDevice(T*& destPtr, const T* sourcePtr, const std::string& name, CLIP_UINT ndof = SCALAR_FIELD)
        {
            cudaMemcpy(destPtr, sourcePtr, ndof * m_domain->domainSize * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaCheckErrors(("copyFromDeviceToDevice failed for " + name).c_str());
        }
        
        template <typename T, size_t N>
        void symbolOnDevice(const T (&symbol)[N], const T* hostPtr, const std::string& name)
        {
            cudaMemcpyToSymbol(symbol, hostPtr, sizeof(T) * N);
            cudaCheckErrors(("cudaMemcpyToSymbol '" + name + "' failed").c_str());
            cudaDeviceSynchronize();
        }
        
        template <typename T>
        void symbolOnDevice(const T& symbol, const T* hostPtr, const std::string& name)
        {
            cudaMemcpyToSymbol(symbol, hostPtr, sizeof(T));
            cudaCheckErrors(("cudaMemcpyToSymbol '" + name + "' failed").c_str());
            cudaDeviceSynchronize();
        }


        template <typename T>
        void writeHostToFile(const T* hostPtr, const std::string& filename, CLIP_UINT size)
        {
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open())
            {
                Logger::Error("Failed to open file for saving: " + filename);
                return;
            }
            file.write(reinterpret_cast<const char*>(hostPtr), size * sizeof(T));
            file.close();
        }
        


        template <typename T>
        void readHostFromFile(T* hostPtr, const std::string& filename, CLIP_UINT size)
        {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open())
            {
                Logger::Error("Failed to open file for reading: " + filename);
                return;
            }
        
            file.read(reinterpret_cast<char*>(hostPtr), size * sizeof(T));
            file.close();
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

        CLIP_REAL *dev_f_prev;
        CLIP_REAL *dev_g_prev;

        };

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




    void createVectors();
    void updateDevice();
    void updateHost();


    private:

        const InputData* m_idata;
        const Domain* m_domain;
        const Boundary* m_boundary;

    };

}
