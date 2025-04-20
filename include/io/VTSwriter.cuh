#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <Solver.cuh>
#include <DataArray.cuh>
#include <Boundary.cuh>
#include <Domain.cuh>


namespace clip
{

    class VTSwriter
    {

    public:
        explicit Solver(InputData idata);

        virtual ~Solver();




        

    private:
    Domain m_domain;


    }



}
