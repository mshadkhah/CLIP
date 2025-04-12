#pragma once
#include <includes.h>
#include <InputData.cuh>

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
        
    
}



