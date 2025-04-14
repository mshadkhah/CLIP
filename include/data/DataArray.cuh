#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <DataTypes.cuh>


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
// #define THREAD_IDX_X (threadIdx.x + blockIdx.x * blockDim.x + 1)
// #define THREAD_IDX_Y (threadIdx.y + blockIdx.y * blockDim.y + 1)
// #define THREAD_IDX_Z (threadIdx.z + blockIdx.z * blockDim.z + 1)


__constant__ CLIP_UINT domainExtent[DIM];
// __constant__ CLIP_UINT domainExtentGhosted[DIM];



namespace clip {

    class DataArray {


        public:
            explicit DataArray(InputData idata);


            template<typename T>
            void allocateOnDevice(T*& devPtr, const char* name, CLIP_UINT ndof = SCALAR) {
                cudaMalloc((void**)&devPtr, ndof * domainSize * sizeof(T));
                cudaCheckErrors(("cudaMalloc '" + std::string(name) + "' fail").c_str());
            }
        
            template <typename T, size_t N>
            void symbolOnDevice(const T (&symbol)[N], const T* hostPtr, const char* name) {
                cudaMemcpyToSymbol(symbol, hostPtr, sizeof(T) * N);
                cudaCheckErrors((std::string("cudaMemcpyToSymbol '") + name + "' failed").c_str());
            }

            template <typename T>
            void symbolOnDevice(const T& symbol, const T* hostPtr, const char* name) {
            cudaMemcpyToSymbol(symbol, hostPtr, sizeof(T));
            cudaCheckErrors((std::string("cudaMemcpyToSymbol '") + name + "' failed").c_str());
            }

            
            template <CLIP_UINT ndof = SCALAR>
            __device__ __forceinline__ static CLIP_UINT getIndex(CLIP_UINT i, CLIP_UINT j, CLIP_UINT k, CLIP_UINT dof = SCALAR) {
                
                return ((i * domainExtent[IDX_Y] + j) * domainExtent[IDX_Z] + k) * ndof + dof;
            }


            template <CLIP_UINT dim, bool ghosted = false>
            __device__ __forceinline__ static bool isInside(CLIP_INT i, CLIP_INT j, CLIP_INT k = 0) {
                constexpr CLIP_UINT offset = ghosted ? 1 : 0;
            
                if constexpr (dim == 2) {
                    return (i >= offset && i < domainExtent[0] - offset) &&
                           (j >= offset && j < domainExtent[1] - offset);
                } else if constexpr (dim == 3) {
                    return (i >= offset && i < domainExtent[0] - offset) &&
                           (j >= offset && j < domainExtent[1] - offset) &&
                           (k >= offset && k < domainExtent[2] - offset);
                } else {
                    return false;
                }
            }
            



        private:

        CLIP_UINT m_nVelocity;
        CLIP_UINT latticeSize;
        CLIP_UINT domainSize;
        CLIP_UINT* m_domainExtent;
        CLIP_UINT* m_domainExtentGhosted;

        protected:



            InputData m_idata;

        

        };
        
    
}



