
#include <equation.cuh>


namespace clip {

    Equation::Equation(InputData idata)
    : m_idata(idata), DataArray(idata){

        m_nVelocity = m_idata.nVelocity;

    }



    Equation::~Equation() {
    }




}

