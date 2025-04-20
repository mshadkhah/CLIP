#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <Solver.cuh>
#include <DataArray.cuh>
#include <Boundary.cuh>
#include <Domain.cuh>
#include "TimeInfo.cuh"


namespace clip
{

    class VTSwriter
    {

    public:
        explicit VTSwriter(const InputData& idata, const Domain& domain, const TimeInfo& ti);

        virtual ~VTSwriter();




        

    private:
    const Domain* m_domain;
    const InputData* m_idata;
    const TimeInfo* m_ti;

    void writeArray();


    }



}
