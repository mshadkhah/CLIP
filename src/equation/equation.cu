
#include <equation.cuh>


namespace clip {

    Equation::Equation(InputData idata)
    : m_idata(idata), DataArray(idata){

        m_nVelocity = m_idata.nVelocity;

    #ifdef ENABLE_2D

        m_domainExtent = new CLIP_UINT[DIM]{m_idata.Nx + 2, m_idata.Ny + 2};
        this->symbolOnDevice(domainExtent, m_domainExtent, "wdomainExtenta");
    
    #elif defined(ENABLE_3D)

        m_domainExtent = new CLIP_UINT[DIM]{m_idata.Nx + 2, m_idata.Ny + 2, m_idata.Nz + 2};
        this->symbolOnDevice(domainExtent, m_domainExtent, "wdomainExtenta");
    
    
    #endif

    }



    Equation::~Equation() {
    }



    __device__ __forceinline__ void Equation::convertD2Q9Weighted(const double in[9], double out[9]) {
        double in0 = in[0], in1 = in[1], in2 = in[2];
        double in3 = in[3], in4 = in[4], in5 = in[5];
        double in6 = in[6], in7 = in[7], in8 = in[8];
    
        out[0] = in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8;
    
        out[1] = -4.0 * in0 - in1 - in2 - in3 - in4 + 2.0 * in5 + 2.0 * in6 + 2.0 * in7 + 2.0 * in8;
    
        out[2] =  4.0 * in0 - 2.0 * in1 - 2.0 * in2 - 2.0 * in3 - 2.0 * in4 + in5 + in6 + in7 + in8;
    
        out[3] =  in1 - in3 + in5 - in6 - in7 + in8;
    
        out[4] = -2.0 * in1 + 2.0 * in3 + in5 - in6 - in7 + in8;
    
        out[5] =  in2 - in4 + in5 + in6 - in7 - in8;
    
        out[6] = -2.0 * in2 + 2.0 * in4 + in5 + in6 - in7 - in8;
    
        out[7] =  in1 - in2 + in3 - in4;
    
        out[8] =  in5 - in6 + in7 - in8;
    }


    __device__ __forceinline__ void Equation::reconvertD2Q9Weighted(const double in[9], double out[9]) {
        double in0 = in[0], in1 = in[1], in2 = in[2];
        double in3 = in[3], in4 = in[4], in5 = in[5];
        double in6 = in[6], in7 = in[7], in8 = in[8];
    
        out[0] = (4.0 * in0 - 4.0 * in1 + 4.0 * in2) / 36.0;
    
        out[1] = (4.0 * in0 - in1 - 2.0 * in2 + 6.0 * in3 - 6.0 * in4 + 9.0 * in7) / 36.0;
    
        out[2] = (4.0 * in0 - in1 - 2.0 * in2 + 6.0 * in5 - 6.0 * in6 - 9.0 * in7) / 36.0;
    
        out[3] = (4.0 * in0 - in1 - 2.0 * in2 - 6.0 * in3 + 6.0 * in4 + 9.0 * in7) / 36.0;
    
        out[4] = (4.0 * in0 - in1 - 2.0 * in2 - 6.0 * in5 + 6.0 * in6 - 9.0 * in7) / 36.0;
    
        out[5] = (4.0 * in0 + 2.0 * in1 + in2 + 6.0 * in3 + 3.0 * in4 + 6.0 * in5 + 3.0 * in6 + 9.0 * in8) / 36.0;
    
        out[6] = (4.0 * in0 + 2.0 * in1 + in2 - 6.0 * in3 - 3.0 * in4 + 6.0 * in5 + 3.0 * in6 - 9.0 * in8) / 36.0;
    
        out[7] = (4.0 * in0 + 2.0 * in1 + in2 - 6.0 * in3 - 3.0 * in4 - 6.0 * in5 - 3.0 * in6 + 9.0 * in8) / 36.0;
    
        out[8] = (4.0 * in0 + 2.0 * in1 + in2 + 6.0 * in3 + 3.0 * in4 - 6.0 * in5 - 3.0 * in6 - 9.0 * in8) / 36.0;
    }
    

}