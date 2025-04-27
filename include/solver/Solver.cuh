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

        template <CLIP_UINT Q>
        void periodicBoundary(CLIP_REAL *dev_a, CLIP_REAL *dev_b = nullptr);
    

        template <CLIP_UINT Q, CLIP_UINT dof>
        void wallBoundary(CLIP_REAL *dev_a, CLIP_REAL *dev_a_post, CLIP_REAL *dev_b = nullptr, CLIP_REAL *dev_b_post = nullptr);

        template <CLIP_UINT Q, CLIP_UINT dof>
        void slipWallBoundary(CLIP_REAL *dev_a, CLIP_REAL *dev_a_post, CLIP_REAL *dev_b = nullptr, CLIP_REAL *dev_b_post = nullptr);

        template <CLIP_UINT Q, CLIP_UINT dof>
        void freeConvectBoundary(CLIP_REAL *dev_vel, CLIP_REAL *dev_a, CLIP_REAL *dev_a_post, CLIP_REAL *dev_b = nullptr, CLIP_REAL *dev_b_post = nullptr);
        
        template <CLIP_UINT Q, CLIP_UINT dof>
        void NeumannBoundary(CLIP_REAL *dev_a, CLIP_REAL *dev_b = nullptr);

        void mirrorBoundary(CLIP_REAL *dev_a);

    private:

    protected:
        const Domain* m_domain;
        const InputData* m_idata;
        const Boundary* m_boundary;
        Domain::DomainInfo m_info;
        Boundary::BCTypeMap m_BCMap;
        WMRT::wallBCMap m_wallBCMap;
        WMRT::slipWallBCMap m_slipWallBCMap;
        DataArray* m_DA;
        dim3 dimGrid, dimBlock;
    };


}
