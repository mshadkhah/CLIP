#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <Boundary.cuh>
#include <Domain.cuh>
#include "DataArray.cuh"


namespace clip
{

    class Solver
    {

    public:
        explicit Solver(const InputData& idata, const Domain& domain, DataArray& DA, const Boundary& boundary);

        virtual ~Solver();

        void flagGenLauncher3();

        template <int Q>
        void periodicBoundary(CLIP_REAL *dev_a, CLIP_REAL *dev_b = nullptr);



    private:


    protected:
        const Domain* m_domain;
        const InputData* m_idata;
        const Boundary* m_boundary;
        Domain::DomainInfo m_info;
        Boundary::BCTypeMap m_BCMap;
        DataArray* m_DA;
        dim3 dimGrid, dimBlock;
    };

}
