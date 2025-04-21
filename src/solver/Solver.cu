#include <Solver.cuh>

namespace clip
{

    Solver::Solver(const InputData& idata, const Domain& domain, DataArray& DA, const Boundary& boundary)
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

                    dev_a[Domain::getIndex<dof>(domain, i, j, domain.ghostDomainMinIdx[IDX_Z], k, q)] = dev_a[Domain::getIndex<dof>(domain, i, j, domain.domainMaxIdx[IDX_Z], q)];
                    dev_a[Domain::getIndex<dof>(domain, i, j, domain.ghostDomainMaxIdx[IDX_Z], k, q)] = dev_a[Domain::getIndex<dof>(domain, i, j, domain.domainMinIdx[IDX_Z], q)];

                    if (dev_b)
                    {
                        dev_b[Domain::getIndex<dof>(domain, i, j, domain.ghostDomainMinIdx[IDX_Z], k, q)] = dev_b[Domain::getIndex<dof>(domain, i, j, domain.domainMaxIdx[IDX_Z], q)];
                        dev_b[Domain::getIndex<dof>(domain, i, j, domain.ghostDomainMaxIdx[IDX_Z], k, q)] = dev_b[Domain::getIndex<dof>(domain, i, j, domain.domainMinIdx[IDX_Z], q)];
                    }
                }

#endif
            }
        }
    }

    template <int Q>
    void Solver::periodicBoundary(CLIP_REAL *dev_a, CLIP_REAL *dev_b)
    {
        if (m_boundary->isPeriodic)
            kernelPeriodicBoundary<Q><<<dimGrid, dimBlock>>>(m_info, m_BCMap, dev_a, dev_b);
    }

    template void clip::Solver::periodicBoundary<9>(CLIP_REAL *, CLIP_REAL *);
    template void clip::Solver::periodicBoundary<19>(CLIP_REAL *, CLIP_REAL *);
    template void clip::Solver::periodicBoundary<1>(CLIP_REAL *, CLIP_REAL *);
}
