#include <Solver.cuh>

namespace clip
{

    Solver::Solver(InputData idata)
        : m_idata(idata), DataArray(idata), m_boundary(idata), m_domain(idata)
    {

#ifdef ENABLE_2D

#elif defined(ENABLE_3D)

#endif

        // this->symbolOnDevice(boundary::s_boundaries, m_idata.boundaries.data(), "boundaries");

        // flagGenLauncher3();
    }

    //     Equation::Equation(InputData idata)
    //     : m_idata(idata), DataArray(idata){

    //         m_nVelocity = m_idata.nVelocity;

    // #ifdef ENABLE_2D
    //         m_ex = new CLIP_INT[WMRT::Q]{0, 1, 0, -1, 0, 1, -1, -1, 1};
    //         m_ey = new CLIP_INT[WMRT::Q]{0, 0, 1, 0, -1, 1, 1, -1, -1};
    //         m_wa = new CLIP_REAL[WMRT::Q]{4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

    //         this->symbolOnDevice(WMRT::ex, m_ex, "ex");
    //         this->symbolOnDevice(WMRT::ey, m_ey, "ey");
    //         this->symbolOnDevice(WMRT::wa, m_wa, "wa");

    // #elif defined(ENABLE_3D)
    //         m_ex = new CLIP_INT[WMRT::Q]{0, 1, 0, -1, 0, 1, -1, -1, 1};
    //         m_ey = new CLIP_INT[WMRT::Q]{0, 0, 1, 0, -1, 1, 1, -1, -1};
    //         m_ez = new CLIP_INT[WMRT::Q]{0, 0, 1, 0, -1, 1, 1, -1, -1};
    //         m_wa = new CLIP_REAL[WMRT::Q]{4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

    //         this->symbolOnDevice(WMRT::ex, m_ex, "ex");
    //         this->symbolOnDevice(WMRT::ey, m_ey, "ey");
    //         this->symbolOnDevice(WMRT::ez, m_ez, "ez");
    //         this->symbolOnDevice(WMRT::wa, m_wa, "wa");

    // #endif

    // // this->symbolOnDevice(boundary::s_boundaries, m_idata.boundaries.data(), "boundaries");

    // flagGenLauncher3();

    //     }

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

        flagGen3<<<dimGrid, dimBlock>>>();
        cudaDeviceSynchronize();
    }

    Solver::~Solver()
    {
    }

    template <int dof = 1>
    __global__ void kernelPeriodicBoundary(const Domain::DomainInfo domain, const Boundary::BCTypeMap BCmap,
                                           CLIP_REAL *dev_a, CLIP_REAL *dev_b = nullptr)
    {
        using namespace boundary;

        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = Domain::getIndex(domain, i, j, k);
        const CLIP_UINT idx_X = Domain::getIndex<DIM>(domain, i, j, k, IDX_X);
        const CLIP_UINT idx_Y = Domain::getIndex<DIM>(domain, i, j, k, IDX_Y);

#ifdef ENABLE_3D
        const CLIP_UINT idx_Z = Domain::getIndex<DIM>(domain, i, j, k, IDX_Z);
#endif

        if (Domain::isInside<DIM>(domain, i, j, k))
        {

#pragma unroll
            for (CLIP_UINT q = 0; q < dof; ++q)
            {

                if (BCmap.types[object::XMinus] == Boundary::Type::Periodic || BCmap.types[object::XPlus] == Boundary::Type::Periodic)
                {
                    dev_a[Domain::getIndex<dof>(domain, domain.ghostDomainMinIdx[IDX_X], j, k, q)] = dev_a[Domain::getIndex<dof>(domain, domain.domainMinIdx[IDX_X], j, k, q)];
                    dev_a[Domain::getIndex<dof>(domain, domain.ghostDomainMaxIdx[IDX_X], j, k, q)] = dev_a[Domain::getIndex<dof>(domain, domain.domainMaxIdx[IDX_X], j, k, q)];

                    if (dev_b)
                    {
                        dev_b[Domain::getIndex<dof>(domain, domain.ghostDomainMinIdx[IDX_X], j, k, q)] = dev_b[Domain::getIndex<dof>(domain, domain.domainMinIdx[IDX_X], j, k, q)];
                        dev_b[Domain::getIndex<dof>(domain, domain.ghostDomainMaxIdx[IDX_X], j, k, q)] = dev_b[Domain::getIndex<dof>(domain, domain.domainMaxIdx[IDX_X], j, k, q)];
                    }
                }

                if (BCmap.types[object::YMinus] == Boundary::Type::Periodic || BCmap.types[object::YPlus] == Boundary::Type::Periodic)
                {

                    dev_a[Domain::getIndex<dof>(domain, i, domain.ghostDomainMinIdx[IDX_Y], k, q)] = dev_a[Domain::getIndex<dof>(domain, i, domain.domainMinIdx[IDX_Y], k, q)];
                    dev_a[Domain::getIndex<dof>(domain, i, domain.ghostDomainMaxIdx[IDX_Y], k, q)] = dev_a[Domain::getIndex<dof>(domain, i, domain.domainMaxIdx[IDX_Y], k, q)];

                    if (dev_b)
                    {
                        dev_b[Domain::getIndex<dof>(domain, i, domain.ghostDomainMinIdx[IDX_Y], k, q)] = dev_b[Domain::getIndex<dof>(domain, i, domain.domainMinIdx[IDX_X], k, q)];
                        dev_b[Domain::getIndex<dof>(domain, i, domain.ghostDomainMaxIdx[IDX_Y], k, q)] = dev_b[Domain::getIndex<dof>(domain, i, domain.domainMaxIdx[IDX_Y], k, q)];
                    }
                }

#ifdef ENABLE_3D

                if (BCmap.types[object::ZMinus] == Boundary::Type::Periodic || BCmap.types[object::ZPlus] == Boundary::Type::Periodic)
                {

                    dev_a[Domain::getIndex<dof>(domain, i, j, domain.ghostDomainMinIdx[IDX_Z], k, q)] = dev_a[Domain::getIndex<dof>(domain, i, j, domain.domainMinIdx[IDX_Z], q)];
                    dev_a[Domain::getIndex<dof>(domain, i, j, domain.ghostDomainMaxIdx[IDX_Z], k, q)] = dev_a[Domain::getIndex<dof>(domain, i, j, domain.domainMaxIdx[IDX_Z], q)];

                    if (dev_b)
                    {
                        dev_b[Domain::getIndex<dof>(domain, i, j, domain.ghostDomainMinIdx[IDX_Z], k, q)] = dev_b[Domain::getIndex<dof>(domain, i, j, domain.domainMinIdx[IDX_Z], q)];
                        dev_b[Domain::getIndex<dof>(domain, i, j, domain.ghostDomainMaxIdx[IDX_Z], k, q)] = dev_b[Domain::getIndex<dof>(domain, i, j, domain.domainMaxIdx[IDX_Z], q)];
                    }
                }

#endif
            }
        }
    }

    template <int Q>
    void Solver::periodicBoundary(double *dev_a, double *dev_b)
    {
        if(m_boundary.isPeriodic)
        kernelPeriodicBoundary<Q><<<DataArray::dimGrid, DataArray::dimBlock>>>(m_domain.info, m_boundary.BCMap, dev_a, dev_b);
    }
}
