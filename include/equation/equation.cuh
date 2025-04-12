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





        private:
            InputData m_idata;
            CLIP_INT m_nVelocity;
            CLIP_INT* m_ex;
            CLIP_INT* m_ey;
            CLIP_INT* m_ez;
            CLIP_REAL* m_wa;



        

        };


        Equation::Equation(InputData idata)
        : m_idata(idata), DataArray(idata){

            m_nVelocity = m_idata.nVelocity;

#ifdef ENABLE_2D
            m_ex = new CLIP_INT[m_nVelocity]{0, 1, 0, -1, 0, 1, -1, -1, 1};
            m_ey = new CLIP_INT[m_nVelocity]{0, 0, 1, 0, -1, 1, 1, -1, -1};
            m_wa = new CLIP_REAL[m_nVelocity]{4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};
#elif defined(ENABLE_3D)
            m_ex = new CLIP_INT[m_nVelocity]{0, 1, 0, -1, 0, 1, -1, -1, 1};
            m_ey = new CLIP_INT[m_nVelocity]{0, 0, 1, 0, -1, 1, 1, -1, -1};
            m_ez = new CLIP_INT[m_nVelocity]{0, 0, 1, 0, -1, 1, 1, -1, -1};
            m_wa = new CLIP_REAL[m_nVelocity]{4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};
#endif

        }

        

        Equation::~Equation() {
            if (m_ex) delete[] m_ex;
            if (m_ey) delete[] m_ey;
            if (m_ez) delete[] m_ez;
            if (m_wa) delete[] m_wa;

        }
        
    
}



