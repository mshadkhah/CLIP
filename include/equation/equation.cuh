#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <equation.cuh>
#include <DataArray.cuh>







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


            


        };


    
}



