#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <Boundary.cuh>
#include <Domain.cuh>
#include "DataArray.cuh"
#include "Geometry.cuh"


namespace clip
{

    class Solver
    {

    public:
        explicit Solver(const InputData& idata, const Domain& domain, DataArray& DA, const Boundary& boundary, const Geometry& geom);

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

        void velocityBoundary(CLIP_REAL *dev_vel, CLIP_REAL *dev_f, CLIP_REAL *dev_g);

        void mirrorBoundary(CLIP_REAL *dev_a);

        __device__ __forceinline__ static CLIP_REAL Equilibrium_new(const WMRT::WMRTvelSet velSet, CLIP_UINT q, CLIP_REAL Ux, CLIP_REAL Uy, CLIP_REAL Uz)
        {
            // using namespace nsAllen;
            const CLIP_INT exq = velSet.ex[q];
            const CLIP_INT eyq = velSet.ey[q];
            const CLIP_REAL waq = velSet.wa[q];
    
    #ifdef ENABLE_2D
            const CLIP_REAL eU = exq * Ux + eyq * Uy;
            const CLIP_REAL U2 = Ux * Ux + Uy * Uy;
    #elif defined(ENABLE_3D)
            const CLIP_INT ezq = velSet.ez[q];
            const CLIP_REAL eU = exq * Ux + eyq * Uy + ezq * Uz;
            const CLIP_REAL U2 = Ux * Ux + Uy * Uy + Uz * Uz;
    #endif
    
            return waq * (3.0 * eU + 4.5 * eU * eU - 1.5 * U2);
        }


    private:

    protected:
        const Domain* m_domain;
        const InputData* m_idata;
        const Boundary* m_boundary;
        const Geometry* m_geom;
        Geometry::GeometryDevice m_geomPool;
        Domain::DomainInfo m_info;
        Boundary::BCTypeMap m_BCMap;
        WMRT::wallBCMap m_wallBCMap;
        WMRT::slipWallBCMap m_slipWallBCMap;
        WMRT::WMRTvelSet m_velSet;
        DataArray* m_DA;
        dim3 dimGrid, dimBlock;
    };


}
