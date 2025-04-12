#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <TimeInfo.cuh>

const double pi = 3.14159265358979L;

__constant__ int ex[9];
__constant__ int ey[9];
__constant__ int N[2];
__constant__ int M[2];
__constant__ double wa[9];


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



namespace clip {

    class DataArray {


        public:
            explicit DataArray(InputData idata);
            void allocateOnDevice(CLIP_REAL* devPtr, const char* name, bool isMacro = false);
            void symbolOnDevice(CLIP_REAL hostVar, CLIP_REAL* devPtr, const char* name, size_t size = 1);


        private:

        CLIP_UINT m_nVelocity;
        CLIP_UINT latticeDimension;
        CLIP_UINT domainDimension;

        protected:



            InputData m_idata;












        

        };


        DataArray::DataArray(InputData idata)
        :m_idata(idata){
            m_nVelocity = m_idata.nVelocity;

#ifdef ENABLE_2D
            domainDimension = (m_idata.Nx + 2) * (m_idata.Ny + 2);
#elif defined(ENABLE_3D)
            domainDimension = (m_idata.Nx + 2) * (m_idata.Ny + 2) * (m_idata.Nz + 2);
#endif
            latticeDimension = domainDimension * m_nVelocity;

        };




        void DataArray::allocateOnDevice(CLIP_REAL* devPtr, const char* name, bool isMacro){
            CLIP_UINT size = 0;
            if (isMacro)
                size = domainDimension;
            else 
                size = latticeDimension;

            cudaMalloc((void **)&devPtr, size * sizeof(CLIP_REAL));
            cudaCheckErrors(("cudaMalloc '" + std::string(name) + "' fail").c_str());
        }


        
        void DataArray::symbolOnDevice(CLIP_REAL hostVar, CLIP_REAL* devPtr, const char* name, size_t size){

            cudaMemcpyToSymbol(hostVar, &devPtr[0], size * sizeof(CLIP_REAL));
            cudaCheckErrors(("cudaMalloc '" + std::string(name) + "' fail").c_str());
        }

        
    
}



