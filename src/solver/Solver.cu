#include <Solver.cuh>

namespace clip
{

    Solver::Solver(const InputData &idata, const Domain &domain, DataArray &DA, const Boundary &boundary)
        : m_idata(&idata), m_domain(&domain), m_DA(&DA), m_boundary(&boundary)
    {

        dimGrid = m_DA->dimGrid;
        dimBlock = m_DA->dimBlock;
        m_info = m_domain->info;
        m_BCMap = m_boundary->BCMap;

#ifdef ENABLE_2D

#elif defined(ENABLE_3D)

#endif

        // this->symbolOnDevice(boundary::s_boundaries, m_idata.boundaries.data(), "boundaries");

        // flagGenLauncher3();
    }

    __global__ void flagGen3()
    {
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        // const CLIP_UINT idx_SCALAR = DataArray::getIndex(i, j, k);

        // printf("Thread index: i = %d, j = %d, k = %d\n", i, j, k);
        // printf("Thread index2: %d \n", idx_SCALAR);

        // if (DataArray::isInside<DIM>(i, j, k)){

        // printf("Thread index: i = %d, j = %d, k = %d\n", i, j, k);
        // printf("index: i = %d\n", DataArray::getDomainExtent(IDX_X));
        // printf("index: s_domainExtent = %d\n", s_domainExtent[IDX_Y]);

        // }
        // printf("index: inside equation getDomainExtent = %d\n", DataArray::getDomainExtent(1));
        // printf("index:  inside equation s_domainExtent = %d\n", s_domainExtent[IDX_Y]);

        // printf("index:  inside equation ex = %d\n", WMRT::ex[3]);
        // printf("index: i = %d\n", idx_SCALAR);
        // printf("Thread index: i = %d, j = %d, k = %d\n", i, j, k);
    }

    void Solver::flagGenLauncher3()
    {

        // flagGen3<<<dimGrid, dimBlock>>>();
        cudaDeviceSynchronize();
    }

    Solver::~Solver()
    {
    }

    template <int dof = 1>
    __global__ void kernelPeriodicBoundary(const Domain::DomainInfo domain, const Boundary::BCTypeMap BCmap,
                                           CLIP_REAL *dev_a, CLIP_REAL *dev_b = nullptr)
    {

        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = Domain::getIndex(domain, i, j, k);

        // printf("i: %d,    j: %d", i, j);

        if (Domain::isInside<DIM>(domain, i, j, k))
        {

            // printf("i = %/d,  j = %d \n", i,j);
#pragma unroll
            for (CLIP_UINT q = 0; q < dof; ++q)
            {

                if (BCmap.types[object::XMinus] == Boundary::Type::Periodic || BCmap.types[object::XPlus] == Boundary::Type::Periodic)
                {
                    dev_a[Domain::getIndex<dof>(domain, domain.ghostDomainMinIdx[IDX_X], j, k, q)] = dev_a[Domain::getIndex<dof>(domain, domain.domainMaxIdx[IDX_X], j, k, q)];
                    dev_a[Domain::getIndex<dof>(domain, domain.ghostDomainMaxIdx[IDX_X], j, k, q)] = dev_a[Domain::getIndex<dof>(domain, domain.domainMinIdx[IDX_X], j, k, q)];

                    // printf("i: %d,    j: %d", i, j);

                    if (dev_b)
                    {
                        dev_b[Domain::getIndex<dof>(domain, domain.ghostDomainMinIdx[IDX_X], j, k, q)] = dev_b[Domain::getIndex<dof>(domain, domain.domainMaxIdx[IDX_X], j, k, q)];
                        dev_b[Domain::getIndex<dof>(domain, domain.ghostDomainMaxIdx[IDX_X], j, k, q)] = dev_b[Domain::getIndex<dof>(domain, domain.domainMinIdx[IDX_X], j, k, q)];
                    }
                }

                if (BCmap.types[object::YMinus] == Boundary::Type::Periodic || BCmap.types[object::YPlus] == Boundary::Type::Periodic)
                {

                    dev_a[Domain::getIndex<dof>(domain, i, domain.ghostDomainMinIdx[IDX_Y], k, q)] = dev_a[Domain::getIndex<dof>(domain, i, domain.domainMaxIdx[IDX_Y], k, q)];
                    dev_a[Domain::getIndex<dof>(domain, i, domain.ghostDomainMaxIdx[IDX_Y], k, q)] = dev_a[Domain::getIndex<dof>(domain, i, domain.domainMinIdx[IDX_Y], k, q)];

                    if (dev_b)
                    {
                        dev_b[Domain::getIndex<dof>(domain, i, domain.ghostDomainMinIdx[IDX_Y], k, q)] = dev_b[Domain::getIndex<dof>(domain, i, domain.domainMaxIdx[IDX_Y], k, q)];
                        dev_b[Domain::getIndex<dof>(domain, i, domain.ghostDomainMaxIdx[IDX_Y], k, q)] = dev_b[Domain::getIndex<dof>(domain, i, domain.domainMinIdx[IDX_Y], k, q)];
                    }
                }

#ifdef ENABLE_3D

                if (BCmap.types[object::ZMinus] == Boundary::Type::Periodic || BCmap.types[object::ZPlus] == Boundary::Type::Periodic)
                {

                    dev_a[Domain::getIndex<dof>(domain, i, j, domain.ghostDomainMinIdx[IDX_Z], q)] = dev_a[Domain::getIndex<dof>(domain, i, j, domain.domainMaxIdx[IDX_Z], q)];
                    dev_a[Domain::getIndex<dof>(domain, i, j, domain.ghostDomainMaxIdx[IDX_Z], q)] = dev_a[Domain::getIndex<dof>(domain, i, j, domain.domainMinIdx[IDX_Z], q)];

                    if (dev_b)
                    {
                        dev_b[Domain::getIndex<dof>(domain, i, j, domain.ghostDomainMinIdx[IDX_Z], q)] = dev_b[Domain::getIndex<dof>(domain, i, j, domain.domainMaxIdx[IDX_Z], q)];
                        dev_b[Domain::getIndex<dof>(domain, i, j, domain.ghostDomainMaxIdx[IDX_Z], q)] = dev_b[Domain::getIndex<dof>(domain, i, j, domain.domainMinIdx[IDX_Z], q)];
                    }
                }

#endif
            }
        }
    }

    template <CLIP_UINT Q, CLIP_UINT dof>
    __global__ void kernelFullBouncBack(const Domain::DomainInfo domain, const Boundary::BCTypeMap BCmap, const WMRT::wallBCMap wallBCMap,
                                        CLIP_REAL *dev_a, CLIP_REAL *dev_b = nullptr)
    {

        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = Domain::getIndex(domain, i, j, k);

        // printf("i: %d,    j: %d", i, j);

        if (Domain::isInside<DIM>(domain, i, j, k))
        {

            // printf("i = %/d,  j = %d \n", i,j);
#pragma unroll
            for (CLIP_UINT q = 0; q < dof; ++q)
            {

                if (BCmap.types[object::XMinus] == Boundary::Type::Wall)
                {
                    dev_a[Domain::getIndex<Q>(domain, domain.ghostDomainMinIdx[IDX_X], j, k, wallBCMap.XMinus[q])] = dev_a[Domain::getIndex<Q>(domain, domain.domainMinIdx[IDX_X], j, k, wallBCMap.XPlus[q])];

                    if (dev_b)
                    {
                        dev_b[Domain::getIndex<Q>(domain, domain.ghostDomainMinIdx[IDX_X], j, k, wallBCMap.XMinus[q])] = dev_b[Domain::getIndex<Q>(domain, domain.domainMinIdx[IDX_X], j, k, wallBCMap.XPlus[q])];
                    }
                }

                if (BCmap.types[object::XPlus] == Boundary::Type::Wall)
                {
                    dev_a[Domain::getIndex<Q>(domain, domain.ghostDomainMaxIdx[IDX_X], j, k, wallBCMap.XPlus[q])] = dev_a[Domain::getIndex<Q>(domain, domain.domainMaxIdx[IDX_X], j, k, wallBCMap.XMinus[q])];

                    if (dev_b)
                    {
                        dev_b[Domain::getIndex<Q>(domain, domain.ghostDomainMaxIdx[IDX_X], j, k, wallBCMap.XPlus[q])] = dev_b[Domain::getIndex<Q>(domain, domain.domainMaxIdx[IDX_X], j, k, wallBCMap.XMinus[q])];
                    }
                }

                if (BCmap.types[object::YMinus] == Boundary::Type::Wall)
                {

                    dev_a[Domain::getIndex<Q>(domain, i, domain.ghostDomainMinIdx[IDX_Y], k, wallBCMap.YMinus[q])] = dev_a[Domain::getIndex<Q>(domain, i, domain.domainMinIdx[IDX_Y], k, wallBCMap.XPlus[q])];

                    if (dev_b)
                    {
                        dev_b[Domain::getIndex<Q>(domain, i, domain.ghostDomainMinIdx[IDX_Y], k, wallBCMap.YMinus[q])] = dev_b[Domain::getIndex<Q>(domain, i, domain.domainMinIdx[IDX_Y], k, wallBCMap.XPlus[q])];
                    }
                }

                if (BCmap.types[object::YPlus] == Boundary::Type::Wall)
                {

                    dev_a[Domain::getIndex<Q>(domain, i, domain.ghostDomainMaxIdx[IDX_Y], k, wallBCMap.YPlus[q])] = dev_a[Domain::getIndex<Q>(domain, i, domain.domainMaxIdx[IDX_Y], k, wallBCMap.YMinus[q])];

                    if (dev_b)
                    {
                        dev_b[Domain::getIndex<Q>(domain, i, domain.ghostDomainMaxIdx[IDX_Y], k, wallBCMap.YPlus[q])] = dev_b[Domain::getIndex<Q>(domain, i, domain.domainMaxIdx[IDX_Y], k, wallBCMap.YMinus[q])];
                    }
                }

#ifdef ENABLE_3D

                if (BCmap.types[object::ZMinus] == Boundary::Type::Wall)
                {

                    dev_a[Domain::getIndex<Q>(domain, i, j, domain.ghostDomainMinIdx[IDX_Z], wallBCMap.ZMinus[q])] = dev_a[Domain::getIndex<Q>(domain, i, j, domain.domainMinIdx[IDX_Z], wallBCMap.ZPlus[q])];

                    if (dev_b)
                    {
                        dev_b[Domain::getIndex<Q>(domain, i, j, domain.ghostDomainMinIdx[IDX_Z], wallBCMap.ZMinus[q])] = dev_b[Domain::getIndex<Q>(domain, i, j, domain.domainMinIdx[IDX_Z], wallBCMap.ZPlus[q])];
                    }
                }

                if (BCmap.types[object::ZPlus] == Boundary::Type::Wall)
                {

                    dev_a[Domain::getIndex<Q>(domain, i, j, domain.ghostDomainMaxIdx[IDX_Z], wallBCMap.ZPlus[q])] = dev_a[Domain::getIndex<Q>(domain, i, j, domain.domainMaxIdx[IDX_Z], wallBCMap.ZMinus[q])];

                    if (dev_b)
                    {
                        dev_b[Domain::getIndex<Q>(domain, i, j, domain.ghostDomainMaxIdx[IDX_Z], wallBCMap.ZPlus[q])] = dev_b[Domain::getIndex<Q>(domain, i, j, domain.domainMaxIdx[IDX_Z], wallBCMap.ZMinus[q])];
                    }
                }

#endif
            }
        }
    }

    template <CLIP_UINT Q, CLIP_UINT dof>
    __global__ void kernelHalfBounceBack(const Domain::DomainInfo domain, const Boundary::BCTypeMap BCmap,
                                         const WMRT::wallBCMap wallMap, CLIP_REAL *dev_a, CLIP_REAL *dev_a_post, CLIP_REAL *dev_b, CLIP_REAL *dev_b_post)
    {
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        if (!Domain::isInside<DIM>(domain, i, j, k))
            return;

#pragma unroll
        for (CLIP_UINT q = 0; q < dof; ++q)
        {
            // X boundaries
            if (BCmap.types[object::XMinus] == Boundary::Type::Wall && i == domain.domainMinIdx[IDX_X])
            {
                const CLIP_UINT idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.XMinus[q]);
                const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.XPlus[q]);
                dev_a_post[idx] = dev_a[opp_idx];
                if (dev_b)
                    dev_b_post[idx] = dev_b[opp_idx];
            }

            if (BCmap.types[object::XPlus] == Boundary::Type::Wall && i == domain.domainMaxIdx[IDX_X])
            {
                const CLIP_UINT idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.XPlus[q]);
                const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.XMinus[q]);
                dev_a_post[idx] = dev_a[opp_idx];
                if (dev_b)
                    dev_b_post[idx] = dev_b[opp_idx];
            }

            // Y boundaries
            if (BCmap.types[object::YMinus] == Boundary::Type::Wall && j == domain.domainMinIdx[IDX_Y])
            {
                const CLIP_UINT idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.YMinus[q]);
                const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.YPlus[q]);
                dev_a_post[idx] = dev_a[opp_idx];
                if (dev_b)
                    dev_b_post[idx] = dev_b[opp_idx];
                    // if (i == 16 && j == 1)
                    // printf ("q = %d  f = %f\n", wallMap.YMinus[q], dev_a[Domain::getIndex<Q>(domain, i, j, k, wallMap.YMinus[q])]);

                    // printf("q= %d, opp= %d \n", wallMap.YMinus[q], wallMap.YPlus[q]);
            }

            if (BCmap.types[object::YPlus] == Boundary::Type::Wall && j == domain.domainMaxIdx[IDX_Y])
            {
                const CLIP_UINT idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.YPlus[q]);
                const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.YMinus[q]);
                dev_a_post[idx] = dev_a[opp_idx];
                if (dev_b)
                    dev_b_post[idx] = dev_b[opp_idx];
            }

#ifdef ENABLE_3D
            // Z boundaries
            if (BCmap.types[object::ZMinus] == Boundary::Type::Wall && k == domain.domainMinIdx[IDX_Z])
            {
                const CLIP_UINT idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.ZMinus[q]);
                const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.ZPlus[q]);
                dev_a_post[idx] = dev_a[opp_idx];
                if (dev_b)
                    dev_b_post[idx] = dev_b[opp_idx];
            }

            if (BCmap.types[object::ZPlus] == Boundary::Type::Wall && k == domain.domainMaxIdx[IDX_Z])
            {
                const CLIP_UINT idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.ZPlus[q]);
                const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.ZMinus[q]);
                dev_a_post[idx] = dev_a[opp_idx];
                if (dev_b)
                    dev_b_post[idx] = dev_b[opp_idx];
            }
#endif
        }
    }


    template <CLIP_UINT Q, CLIP_UINT dof>
    __global__ void kernelBounceBackEdgeCorrection(const Domain::DomainInfo domain, const Boundary::BCTypeMap BCmap,
                                         const WMRT::wallBCMap wallMap, CLIP_REAL *dev_a, CLIP_REAL *dev_a_post, CLIP_REAL *dev_b, CLIP_REAL *dev_b_post)
    {
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        if (!Domain::isInside<DIM>(domain, i, j, k))
            return;

#pragma unroll
        for (CLIP_UINT q = 0; q < dof; ++q)
        {
            // X boundaries
            if (BCmap.types[object::XMinus] == Boundary::Type::Wall ||  BCmap.types[object::YMinus] == Boundary::Type::Wall ||  BCmap.types[object::ZMinus] == Boundary::Type::Wall)
            {
                const CLIP_UINT idx = Domain::getIndex<Q>(domain, domain.domainMinIdx[IDX_X], domain.domainMinIdx[IDX_Y], domain.domainMinIdx[IDX_Z], wallMap.XMinus[q]);
                const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, domain.domainMinIdx[IDX_X], domain.domainMinIdx[IDX_Y], domain.domainMinIdx[IDX_Z], wallMap.XPlus[q]);
                dev_a_post[idx] = dev_a[opp_idx];
                if (dev_b)
                    dev_b_post[idx] = dev_b[opp_idx];
            }

            if (BCmap.types[object::XPlus] == Boundary::Type::Wall && i == domain.domainMaxIdx[IDX_X])
            {
                const CLIP_UINT idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.XPlus[q]);
                const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.XMinus[q]);
                dev_a_post[idx] = dev_a[opp_idx];
                if (dev_b)
                    dev_b_post[idx] = dev_b[opp_idx];
            }

            // Y boundaries
            if (BCmap.types[object::YMinus] == Boundary::Type::Wall && j == domain.domainMinIdx[IDX_Y])
            {
                const CLIP_UINT idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.YMinus[q]);
                const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.YPlus[q]);
                dev_a_post[idx] = dev_a[opp_idx];
                if (dev_b)
                    dev_b_post[idx] = dev_b[opp_idx];
                    // if (i == 16 && j == 1)
                    // printf ("q = %d  f = %f\n", wallMap.YMinus[q], dev_a[Domain::getIndex<Q>(domain, i, j, k, wallMap.YMinus[q])]);

                    // printf("q= %d, opp= %d \n", wallMap.YMinus[q], wallMap.YPlus[q]);
            }

            if (BCmap.types[object::YPlus] == Boundary::Type::Wall && j == domain.domainMaxIdx[IDX_Y])
            {
                const CLIP_UINT idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.YPlus[q]);
                const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.YMinus[q]);
                dev_a_post[idx] = dev_a[opp_idx];
                if (dev_b)
                    dev_b_post[idx] = dev_b[opp_idx];
            }

#ifdef ENABLE_3D
            // Z boundaries
            if (BCmap.types[object::ZMinus] == Boundary::Type::Wall && k == domain.domainMinIdx[IDX_Z])
            {
                const CLIP_UINT idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.ZMinus[q]);
                const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.ZPlus[q]);
                dev_a_post[idx] = dev_a[opp_idx];
                if (dev_b)
                    dev_b_post[idx] = dev_b[opp_idx];
            }

            if (BCmap.types[object::ZPlus] == Boundary::Type::Wall && k == domain.domainMaxIdx[IDX_Z])
            {
                const CLIP_UINT idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.ZPlus[q]);
                const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.ZMinus[q]);
                dev_a_post[idx] = dev_a[opp_idx];
                if (dev_b)
                    dev_b_post[idx] = dev_b[opp_idx];
            }
#endif
        }
    }



    __global__ void kernelMirrorBoundary(const Domain::DomainInfo domain, const Boundary::BCTypeMap BCmap, CLIP_REAL *dev_a)
    {
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        if (Domain::isInside<DIM>(domain, i, j, k))
        {
            const CLIP_UINT idx = Domain::getIndex(domain, i, j, k);

            // XMIN
            if (BCmap.types[object::XMinus] == Boundary::Type::Wall && i == domain.domainMinIdx[IDX_X])
            {
                const CLIP_UINT ghost = Domain::getIndex(domain, domain.ghostDomainMinIdx[IDX_X], j, k);
                dev_a[ghost] = dev_a[idx];
            }

            // XMAX
            if (BCmap.types[object::XPlus] == Boundary::Type::Wall && i == domain.domainMaxIdx[IDX_X])
            {
                const CLIP_UINT ghost = Domain::getIndex(domain, domain.ghostDomainMaxIdx[IDX_X], j, k);
                dev_a[ghost] = dev_a[idx];
            }

            // YMIN
            if (BCmap.types[object::YMinus] == Boundary::Type::Wall && j == domain.domainMinIdx[IDX_Y])
            {
                const CLIP_UINT ghost = Domain::getIndex(domain, i, domain.ghostDomainMinIdx[IDX_Y], k);
                dev_a[ghost] = dev_a[idx];
            }

            // YMAX
            if (BCmap.types[object::YPlus] == Boundary::Type::Wall && j == domain.domainMaxIdx[IDX_Y])
            {
                const CLIP_UINT ghost = Domain::getIndex(domain, i, domain.ghostDomainMaxIdx[IDX_Y], k);
                dev_a[ghost] = dev_a[idx];
            }

#ifdef ENABLE_3D
            // ZMIN
            if (BCmap.types[object::ZMinus] == Boundary::Type::Wall && k == domain.domainMinIdx[IDX_Z])
            {
                const CLIP_UINT ghost = Domain::getIndex(domain, i, j, domain.ghostDomainMinIdx[IDX_Z]);
                dev_a[ghost] = dev_a[idx];
            }

            // ZMAX
            if (BCmap.types[object::ZPlus] == Boundary::Type::Wall && k == domain.domainMaxIdx[IDX_Z])
            {
                const CLIP_UINT ghost = Domain::getIndex(domain, i, j, domain.ghostDomainMaxIdx[IDX_Z]);
                dev_a[ghost] = dev_a[idx];
            }
#endif
        }
    }




    template <CLIP_UINT Q>
    void Solver::periodicBoundary(CLIP_REAL *dev_a, CLIP_REAL *dev_b)
    {
        if (m_boundary->isPeriodic)
            kernelPeriodicBoundary<Q><<<dimGrid, dimBlock>>>(m_info, m_BCMap, dev_a, dev_b);
    }

    template <CLIP_UINT Q, CLIP_UINT dof>
    void Solver::wallBoundary(CLIP_REAL *dev_a, CLIP_REAL *dev_a_post, CLIP_REAL *dev_b, CLIP_REAL *dev_b_post)
    {
        if (m_boundary->isWall)
        kernelHalfBounceBack<Q, dof><<<dimGrid, dimBlock>>>(m_info, m_BCMap, m_wallBCMap, dev_a, dev_a_post, dev_b, dev_b_post);

    }

    void Solver::mirrorBoundary(CLIP_REAL *dev_a)
    {
        if (m_boundary->isWall)
        kernelMirrorBoundary<<<dimGrid, dimBlock>>>(m_info, m_BCMap, dev_a);
    }



    template void clip::Solver::periodicBoundary<9>(CLIP_REAL *, CLIP_REAL *);
    template void clip::Solver::periodicBoundary<19>(CLIP_REAL *, CLIP_REAL *);
    template void clip::Solver::periodicBoundary<1>(CLIP_REAL *, CLIP_REAL *);

    template void clip::Solver::wallBoundary<9,3>(CLIP_REAL *, CLIP_REAL *,CLIP_REAL *, CLIP_REAL *);
    template void clip::Solver::wallBoundary<19,5>(CLIP_REAL *, CLIP_REAL *,CLIP_REAL *, CLIP_REAL *);

}
