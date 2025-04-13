#pragma once
#include <includes.h>
#include <InputData.cuh>


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





    

#define THREAD_IDX_X_GHOSTED (threadIdx.x + blockIdx.x * blockDim.x)
#define THREAD_IDX_Y_GHOSTED (threadIdx.y + blockIdx.y * blockDim.y)
#define THREAD_IDX_Z_GHOSTED (threadIdx.z + blockIdx.z * blockDim.z)
#define THREAD_IDX_X (threadIdx.x + blockIdx.x * blockDim.x + 1)
#define THREAD_IDX_Y (threadIdx.y + blockIdx.y * blockDim.y + 1)
#define THREAD_IDX_Z (threadIdx.z + blockIdx.z * blockDim.z + 1)




namespace clip {

    class DataArray {


        public:
            explicit DataArray(InputData idata);


            template<typename T>
            void allocateOnDevice(T*& devPtr, const char* name, bool isMacro = false) {
                CLIP_UINT size = isMacro ? domainDimension : latticeDimension;
                cudaMalloc((void**)&devPtr, size * sizeof(T));
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

            
            template <size_t N, size_t M = 1>
            __device__ __forceinline__ int getIndex(int i, int j, int k = 0) {
                return (i * N + j) * M + k;
            }



        private:

        CLIP_UINT m_nVelocity;
        CLIP_UINT latticeDimension;
        CLIP_UINT domainDimension;

        protected:



            InputData m_idata;

        

        };
        
    
}



