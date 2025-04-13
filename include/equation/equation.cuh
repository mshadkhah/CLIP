#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <equation.cuh>
#include <DataArray.cuh>


static constexpr CLIP_UINT MAX_Q = 32;
__constant__ CLIP_UINT domainExtent[DIM];
__constant__ CLIP_INT ex[MAX_Q];
__constant__ CLIP_INT ey[MAX_Q];
#ifdef ENABLE_3D
__constant__ CLIP_UINT ez[MAX_Q];
#endif
__constant__ CLIP_REAL wa[MAX_Q];







namespace clip {

    class Equation : public DataArray{

        public:
            explicit Equation(InputData idata);
            virtual ~Equation();





            



            __device__ __forceinline__ void convertD2Q9Weighted(const double in[9], double out[9]);
            __device__ __forceinline__ void reconvertD2Q9Weighted(const double in[9], double out[9]);
            









        private:
            InputData m_idata;
            size_t m_nVelocity;
            CLIP_UINT* m_domainExtent;

            


        };


    
}



