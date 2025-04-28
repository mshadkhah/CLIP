#include <Solver.cuh>

namespace clip
{

    Solver::Solver(const InputData &idata, const Domain &domain, DataArray &DA, const Boundary &boundary, const Geometry &geom)
        : m_idata(&idata), m_domain(&domain), m_DA(&DA), m_boundary(&boundary), m_geom(&geom)
    {

        dimGrid = m_DA->dimGrid;
        dimBlock = m_DA->dimBlock;
        m_info = m_domain->info;
        m_BCMap = m_boundary->BCMap;
        m_geomPool = m_geom->getDeviceStruct();

#ifdef ENABLE_2D

#elif defined(ENABLE_3D)

#endif
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

    template <CLIP_UINT Q, CLIP_UINT dof, typename T>
    __global__ void kernelHalfBounceBack(const Domain::DomainInfo domain, const Boundary::BCTypeMap BCmap,
                                         const T wallMap, CLIP_REAL *dev_a, CLIP_REAL *dev_a_post, CLIP_REAL *dev_b, CLIP_REAL *dev_b_post)
    {
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        if (Domain::isInside<DIM>(domain, i, j, k))
        {
#pragma unroll
            for (CLIP_UINT q = 0; q < dof; ++q)
            {
                // X boundaries
                if ((BCmap.types[object::XMinus] == Boundary::Type::Wall ||
                     BCmap.types[object::XMinus] == Boundary::Type::SlipWall) &&
                    i == domain.domainMinIdx[IDX_X])
                {
                    const CLIP_UINT idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.XMinus[q]);
                    const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.XPlus[q]);
                    dev_a_post[idx] = dev_a[opp_idx];
                    if (dev_b)
                        dev_b_post[idx] = dev_b[opp_idx];
                }

                if ((BCmap.types[object::XPlus] == Boundary::Type::Wall ||
                     BCmap.types[object::XPlus] == Boundary::Type::SlipWall) &&
                    i == domain.domainMaxIdx[IDX_X])
                {
                    const CLIP_UINT idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.XPlus[q]);
                    const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.XMinus[q]);
                    dev_a_post[idx] = dev_a[opp_idx];
                    if (dev_b)
                        dev_b_post[idx] = dev_b[opp_idx];
                }

                // Y boundaries
                if ((BCmap.types[object::YMinus] == Boundary::Type::Wall ||
                     BCmap.types[object::YMinus] == Boundary::Type::SlipWall) &&
                    j == domain.domainMinIdx[IDX_Y])
                {
                    const CLIP_UINT idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.YMinus[q]);
                    const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.YPlus[q]);
                    dev_a_post[idx] = dev_a[opp_idx];
                    if (dev_b)
                        dev_b_post[idx] = dev_b[opp_idx];
                }

                if ((BCmap.types[object::YPlus] == Boundary::Type::Wall ||
                     BCmap.types[object::YPlus] == Boundary::Type::SlipWall) &&
                    j == domain.domainMaxIdx[IDX_Y])
                {
                    const CLIP_UINT idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.YPlus[q]);
                    const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.YMinus[q]);
                    dev_a_post[idx] = dev_a[opp_idx];
                    if (dev_b)
                        dev_b_post[idx] = dev_b[opp_idx];
                }

#ifdef ENABLE_3D
                // Z boundaries
                if ((BCmap.types[object::ZMinus] == Boundary::Type::Wall ||
                     BCmap.types[object::ZMinus] == Boundary::Type::SlipWall) &&
                    k == domain.domainMinIdx[IDX_Z])
                {
                    const CLIP_UINT idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.ZMinus[q]);
                    const CLIP_UINT opp_idx = Domain::getIndex<Q>(domain, i, j, k, wallMap.ZPlus[q]);
                    dev_a_post[idx] = dev_a[opp_idx];
                    if (dev_b)
                        dev_b_post[idx] = dev_b[opp_idx];
                }

                if ((BCmap.types[object::ZPlus] == Boundary::Type::Wall ||
                     BCmap.types[object::ZPlus] == Boundary::Type::SlipWall) &&
                    k == domain.domainMaxIdx[IDX_Z])
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
    }

    template <CLIP_UINT Q, CLIP_UINT dof, typename T>
    __global__ void kernelFreeConvect(const Domain::DomainInfo domain, const Boundary::BCTypeMap BCmap,
                                      const T wallMap, CLIP_REAL *dev_vel,
                                      CLIP_REAL *dev_a, CLIP_REAL *dev_a_prev,
                                      CLIP_REAL *dev_b, CLIP_REAL *dev_b_prev)
    {
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        if (Domain::isInside<DIM>(domain, i, j, k))
        {

            // XMIN
            if (BCmap.types[object::XMinus] == Boundary::Type::FreeConvect && i == domain.domainMinIdx[IDX_X])
            {
#pragma unroll
                for (CLIP_UINT q = 0; q < dof; ++q)
                {
                    const CLIP_UINT idxBoundary = Domain::getIndex<Q>(domain, i, j, k, wallMap.XMinus[q]);
                    const CLIP_UINT idxInterior = Domain::getIndex<Q>(domain, i + 1, j, k, wallMap.XMinus[q]);
                    const CLIP_REAL convectVel = fabs(dev_vel[Domain::getIndex<DIM>(domain, i + 1, j, k, IDX_X)]);

                    dev_a[idxBoundary] = (dev_a_prev[idxBoundary] + convectVel * dev_a[idxInterior]) / (1.0 + convectVel);
                    if (dev_b)
                        dev_b[idxBoundary] = (dev_b_prev[idxBoundary] + convectVel * dev_b[idxInterior]) / (1.0 + convectVel);
                }
            }

            // XMAX
            if (BCmap.types[object::XPlus] == Boundary::Type::FreeConvect && i == domain.domainMaxIdx[IDX_X])
            {
#pragma unroll
                for (CLIP_UINT q = 0; q < dof; ++q)
                {
                    const CLIP_UINT idxBoundary = Domain::getIndex<Q>(domain, i, j, k, wallMap.XPlus[q]);
                    const CLIP_UINT idxInterior = Domain::getIndex<Q>(domain, i - 1, j, k, wallMap.XPlus[q]);
                    const CLIP_REAL convectVel = fabs(dev_vel[Domain::getIndex<DIM>(domain, i - 1, j, k, IDX_X)]);

                    dev_a[idxBoundary] = (dev_a_prev[idxBoundary] + convectVel * dev_a[idxInterior]) / (1.0 + convectVel);
                    if (dev_b)
                        dev_b[idxBoundary] = (dev_b_prev[idxBoundary] + convectVel * dev_b[idxInterior]) / (1.0 + convectVel);
                }
            }

            // YMIN
            if (BCmap.types[object::YMinus] == Boundary::Type::FreeConvect && j == domain.domainMinIdx[IDX_Y])
            {
#pragma unroll
                for (CLIP_UINT q = 0; q < dof; ++q)
                {
                    const CLIP_UINT idxBoundary = Domain::getIndex<Q>(domain, i, j, k, wallMap.YMinus[q]);
                    const CLIP_UINT idxInterior = Domain::getIndex<Q>(domain, i, j + 1, k, wallMap.YMinus[q]);
                    const CLIP_REAL convectVel = fabs(dev_vel[Domain::getIndex<DIM>(domain, i, j + 1, k, IDX_Y)]);

                    dev_a[idxBoundary] = (dev_a_prev[idxBoundary] + convectVel * dev_a[idxInterior]) / (1.0 + convectVel);
                    if (dev_b)
                        dev_b[idxBoundary] = (dev_b_prev[idxBoundary] + convectVel * dev_b[idxInterior]) / (1.0 + convectVel);
                }
            }

            // YMAX
            if (BCmap.types[object::YPlus] == Boundary::Type::FreeConvect && j == domain.domainMaxIdx[IDX_Y])
            {
#pragma unroll
                for (CLIP_UINT q = 0; q < dof; ++q)
                {
                    const CLIP_UINT idxBoundary = Domain::getIndex<Q>(domain, i, j, k, wallMap.YPlus[q]);
                    const CLIP_UINT idxInterior = Domain::getIndex<Q>(domain, i, j - 1, k, wallMap.YPlus[q]);
                    const CLIP_REAL convectVel = fabs(dev_vel[Domain::getIndex<DIM>(domain, i, j - 1, k, IDX_Y)]);

                    dev_a[idxBoundary] = (dev_a_prev[idxBoundary] + convectVel * dev_a[idxInterior]) / (1.0 + convectVel);
                    if (dev_b)
                        dev_b[idxBoundary] = (dev_b_prev[idxBoundary] + convectVel * dev_b[idxInterior]) / (1.0 + convectVel);
                }
            }

#ifdef ENABLE_3D
            // ZMIN
            if (BCmap.types[object::ZMinus] == Boundary::Type::FreeConvect && k == domain.domainMinIdx[IDX_Z])
            {
#pragma unroll
                for (CLIP_UINT q = 0; q < dof; ++q)
                {
                    const CLIP_UINT idxBoundary = Domain::getIndex<Q>(domain, i, j, k, wallMap.ZMinus[q]);
                    const CLIP_UINT idxInterior = Domain::getIndex<Q>(domain, i, j, k + 1, wallMap.ZMinus[q]);
                    const CLIP_REAL convectVel = fabs(dev_vel[Domain::getIndex<DIM>(domain, i, j, k + 1, IDX_Z)]);

                    dev_a[idxBoundary] = (dev_a_prev[idxBoundary] + convectVel * dev_a[idxInterior]) / (1.0 + convectVel);
                    if (dev_b)
                        dev_b[idxBoundary] = (dev_b_prev[idxBoundary] + convectVel * dev_b[idxInterior]) / (1.0 + convectVel);
                }
            }

            // ZMAX
            if (BCmap.types[object::ZPlus] == Boundary::Type::FreeConvect && k == domain.domainMaxIdx[IDX_Z])
            {
#pragma unroll
                for (CLIP_UINT q = 0; q < dof; ++q)
                {
                    const CLIP_UINT idxBoundary = Domain::getIndex<Q>(domain, i, j, k, wallMap.ZPlus[q]);
                    const CLIP_UINT idxInterior = Domain::getIndex<Q>(domain, i, j, k - 1, wallMap.ZPlus[q]);
                    const CLIP_REAL convectVel = fabs(dev_vel[Domain::getIndex<DIM>(domain, i, j, k - 1, IDX_Z)]);

                    dev_a[idxBoundary] = (dev_a_prev[idxBoundary] + convectVel * dev_a[idxInterior]) / (1.0 + convectVel);
                    if (dev_b)
                        dev_b[idxBoundary] = (dev_b_prev[idxBoundary] + convectVel * dev_b[idxInterior]) / (1.0 + convectVel);
                }
            }
#endif
        }
    }

    template <CLIP_UINT Q, CLIP_UINT dof, typename T>
    __global__ void kernelNeumann(const Domain::DomainInfo domain, const Boundary::BCTypeMap BCmap,
                                  const T wallMap, CLIP_REAL *dev_a, CLIP_REAL *dev_b)
    {
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        if (Domain::isInside<DIM>(domain, i, j, k))
        {
#pragma unroll
            for (CLIP_UINT q = 0; q < dof; ++q)
            {

                // X boundaries
                if (BCmap.types[object::XMinus] == Boundary::Type::Neumann && i == domain.domainMinIdx[IDX_X])
                {
                    const CLIP_UINT idxBoundary = Domain::getIndex<Q>(domain, i, j, k, wallMap.XMinus[q]);
                    const CLIP_UINT idxInterior = Domain::getIndex<Q>(domain, i + 1, j, k, wallMap.XMinus[q]);

                    dev_a[idxBoundary] = dev_a[idxInterior];
                    if (dev_b)
                        dev_b[idxBoundary] = dev_b[idxInterior];
                }

                if (BCmap.types[object::XPlus] == Boundary::Type::Neumann && i == domain.domainMaxIdx[IDX_X])
                {
                    const CLIP_UINT idxBoundary = Domain::getIndex<Q>(domain, i, j, k, wallMap.XPlus[q]);
                    const CLIP_UINT idxInterior = Domain::getIndex<Q>(domain, i - 1, j, k, wallMap.XPlus[q]);

                    dev_a[idxBoundary] = dev_a[idxInterior];
                    if (dev_b)
                        dev_b[idxBoundary] = dev_b[idxInterior];
                }

                // Y boundaries
                if (BCmap.types[object::YMinus] == Boundary::Type::Neumann && j == domain.domainMinIdx[IDX_Y])
                {
                    const CLIP_UINT idxBoundary = Domain::getIndex<Q>(domain, i, j, k, wallMap.YMinus[q]);
                    const CLIP_UINT idxInterior = Domain::getIndex<Q>(domain, i, j + 1, k, wallMap.YMinus[q]);

                    dev_a[idxBoundary] = dev_a[idxInterior];
                    if (dev_b)
                        dev_b[idxBoundary] = dev_b[idxInterior];
                }

                if (BCmap.types[object::YPlus] == Boundary::Type::Neumann && j == domain.domainMaxIdx[IDX_Y])
                {
                    const CLIP_UINT idxBoundary = Domain::getIndex<Q>(domain, i, j, k, wallMap.YPlus[q]);
                    const CLIP_UINT idxInterior = Domain::getIndex<Q>(domain, i, j - 1, k, wallMap.YPlus[q]);

                    dev_a[idxBoundary] = dev_a[idxInterior];
                    if (dev_b)
                        dev_b[idxBoundary] = dev_b[idxInterior];
                }

#ifdef ENABLE_3D
                // Z boundaries
                if (BCmap.types[object::ZMinus] == Boundary::Type::Neumann && k == domain.domainMinIdx[IDX_Z])
                {
                    const CLIP_UINT idxBoundary = Domain::getIndex<Q>(domain, i, j, k, wallMap.ZMinus[q]);
                    const CLIP_UINT idxInterior = Domain::getIndex<Q>(domain, i, j, k + 1, wallMap.ZMinus[q]);

                    dev_a[idxBoundary] = dev_a[idxInterior];
                    if (dev_b)
                        dev_b[idxBoundary] = dev_b[idxInterior];
                }

                if (BCmap.types[object::ZPlus] == Boundary::Type::Neumann && k == domain.domainMaxIdx[IDX_Z])
                {
                    const CLIP_UINT idxBoundary = Domain::getIndex<Q>(domain, i, j, k, wallMap.ZPlus[q]);
                    const CLIP_UINT idxInterior = Domain::getIndex<Q>(domain, i, j, k - 1, wallMap.ZPlus[q]);

                    dev_a[idxBoundary] = dev_a[idxInterior];
                    if (dev_b)
                        dev_b[idxBoundary] = dev_b[idxInterior];
                }
#endif
            }
        }
    }

    /////////////////////////////////////////////////////// start

    // template <typname T, typename Q, typename R, typename S>
    __global__ void JetBoundary(const Domain::DomainInfo domain, const Geometry::GeometryDevice geom, const Boundary::BCTypeMap BCmap,
                                const WMRT::WMRTvelSet velSet, const WMRT::slipWallBCMap wallBCMap, CLIP_REAL *dev_c, CLIP_REAL *dev_f, CLIP_REAL *dev_g)
    {
        // const WMRT::velocityBCMap velocityBCMap
        float feq[19], geq[19];
        // int FStop[10] = {4, 8, 9, 16, 18, 3, 10, 7, 17, 15};


        const CLIP_UINT Q = velSet.Q;


        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = Domain::getIndex(domain, i, j, k);

        const CLIP_REAL x = static_cast<CLIP_REAL>(i);
        const CLIP_REAL y = static_cast<CLIP_REAL>(j);
        const CLIP_REAL z = (DIM == 3) ? static_cast<CLIP_REAL>(k) : 0.0;


        if (Domain::isInside<DIM>(domain, i, j, k))
        
        {


            // if (BCmap.types[object::YPlus] == Boundary::Type::Velocity && i == domain.ghostDomainMaxIdx[IDX_X])
            if (j == domain.ghostDomainMaxIdx[IDX_Y])
            {

                if (Geometry::sdf(geom, 0, x, y, z) <= 0)
                {

                    CLIP_REAL Mx = 0;
                    CLIP_REAL Mz = 0;
                    CLIP_REAL N = 0;
#pragma unroll
                    for (int q = 1; q < 19; q++)
                    {

                        const CLIP_REAL fa_wa = Solver::Equilibrium_new(velSet, q, BCmap.val[object::YPlus][IDX_X],
                                                                         BCmap.val[object::YPlus][IDX_Y], BCmap.val[object::YPlus][IDX_Z]);

                                                                         
                        feq[q] = 0.0 * velSet.wa[q] + fa_wa;
                        geq[q] = dev_c[idx_SCALAR] * (fa_wa + velSet.wa[q]);

                        if (velSet.ey[q] == 0)
                        {
                            Mx += velSet.ex[q] * (dev_f[Domain::getIndex<Q>(domain, i, j, k, q)] - feq[q]);
                            N += (dev_g[Domain::getIndex<Q>(domain, i, j, k, q)] - geq[q]);
                        }

                        if (velSet.ex[q] == 0)
                            Mz += velSet.ez[q] * (dev_f[Domain::getIndex<Q>(domain, i, j, k, q)] - feq[q]);
                    }

                    for (int q = 0; q < 5; q++)
                    {

                        const CLIP_UINT idxF = Domain::getIndex<Q>(domain, i, j, k, wallBCMap.YPlus[q]);
                        const CLIP_UINT oppos_idxF = Domain::getIndex<Q>(domain, i, j, k, wallBCMap.YMinus[q]);
                        const CLIP_UINT idx = wallBCMap.YPlus[q];
                        const CLIP_UINT oppos_idx = wallBCMap.YMinus[q];

                        dev_f[idxF] = dev_f[oppos_idxF] + feq[idx] - feq[oppos_idx] - (1.0 / 4.0) * (velSet.ex[idx] * Mx + velSet.ez[idx] * Mz);

                        // dev_g[getIndexf(i, j, k, 4)] = geq[4] + dev_g[getIndexf(i, j, k, 3)] - geq[3];
                        // dev_g[getIndexf(i, j, k, 8)] = dev_g[getIndexf(i, j, k, 7)] + geq[8] - geq[7] - (1.0 / 4.0) * (ex[8] * Mx + ez[8] * Mz);
                        // dev_g[getIndexf(i, j, k, 9)] = dev_g[getIndexf(i, j, k, 10)] + geq[9] - geq[10] - (1.0 / 4.0) * (ex[9] * Mx + ez[9] * Mz);
                        // dev_g[getIndexf(i, j, k, 16)] = dev_g[getIndexf(i, j, k, 15)] + geq[16] - geq[15] - (1.0 / 4.0) * (ex[16] * Mx + ez[16] * Mz);
                        // dev_g[getIndexf(i, j, k, 18)] = dev_g[getIndexf(i, j, k, 17)] + geq[18] - geq[17] - (1.0 / 4.0) * (ex[18] * Mx + ez[18] * Mz);

                        if (q == 0)
                            dev_g[idxF] = geq[idx] - (dev_g[oppos_idxF] - geq[oppos_idx]);

                        else
                            dev_g[idxF] = geq[idx] - (dev_g[oppos_idxF] - geq[oppos_idx]) - N / 4.0;

                        // dev_h[getIndexf(i, j, k, 4)] = heq[4] - (dev_h[getIndexf(i, j, k, 3)] - heq[3]);
                        // dev_h[getIndexf(i, j, k, 8)] = heq[8] - (dev_h[getIndexf(i, j, k, 7)] - heq[7]) - N / 4.0;
                        // dev_h[getIndexf(i, j, k, 9)] = heq[9] - (dev_h[getIndexf(i, j, k, 10)] - heq[10]) - N / 4.0;
                        // dev_h[getIndexf(i, j, k, 16)] = heq[16] - (dev_h[getIndexf(i, j, k, 15)] - heq[15]) - N / 4.0;
                        // dev_h[getIndexf(i, j, k, 18)] = heq[18] - (dev_h[getIndexf(i, j, k, 17)] - heq[17]) - N / 4.0;
                    }
                }
                else
                {
                    for (int q = 0; q < 5; q++)
                    {

                        dev_g[ Domain::getIndex<Q>(domain, i, domain.ghostDomainMaxIdx[IDX_Y], k, wallBCMap.YPlus[q])] = dev_g[ Domain::getIndex<Q>(domain, i, domain.domainMaxIdx[IDX_Y], k, wallBCMap.YMinus[q])];
                        dev_f[ Domain::getIndex<Q>(domain, i, domain.ghostDomainMaxIdx[IDX_Y], k, wallBCMap.YPlus[q])] = dev_f[ Domain::getIndex<Q>(domain, i, domain.domainMaxIdx[IDX_Y], k, wallBCMap.YMinus[q])];
                    }
                }
            }

//             float R = (i - (*dev_x0 + 0.50)) * (i - (*dev_x0 + 0.50)) + (k - (*dev_z0 + 0.50)) * (k - (*dev_z0 + 0.50));

//             if (R < (*dev_r0 * *dev_r0))
//             {

//                 dev_uy[index] = *dev_u0;
//                 dev_ux[index] = 0;
//                 dev_uz[index] = 0;

//                 // dev_c[index] = 1.0L;
//                 dev_p[index] = 0.0L;

// #pragma unroll
//                 for (int q = 0; q < 19; q++)
//                 {
//                     // index = getIndex(i ,j ,k);
//                     float ga_wa = Equilibrium_new(0, *dev_u0, 0, q);
//                     geq[q] = 0.0 * wa[q] + ga_wa;

//                     // float eF = (4.0L * (dev_c[index] * (1.0L - dev_c[index])) * wa[q] * (ex[q] * dev_ni[index] + ey[q] * dev_nj[index] + ez[q] * dev_nk[index])) / *dev_w;
//                     heq[q] = dev_c[index] * (ga_wa + wa[q]);
//                 }

//                 dev_h[getIndexf(i, j, k, 0)] = heq[0];
//                 dev_g[getIndexf(i, j, k, 0)] = geq[0];

//                 CLIP_REAL Mx = 0;
//                 CLIP_REAL My = 0;
//                 CLIP_REAL Mz = 0;

// #pragma unroll
//                 for (int q = 0; q < 19; q++)
//                 {
//                     if (ey[0] == 0)
//                         Mx += ex[q] * (dev_g[getIndexf(i, j, k, q)] - geq[q]);

//                     if (ex[0] == 0)
//                         My += ey[q] * (dev_g[getIndexf(i, j, k, q)] - geq[q]);

//                     if (ex[0] == 0)
//                         Mz += ey[q] * (dev_g[getIndexf(i, j, k, q)] - geq[q]);
//                 }

//                 float Mx = (dev_g[getIndexf(i, j, k, 1)] - geq[1]) + (dev_g[getIndexf(i, j, k, 11)] - geq[11]) + (dev_g[getIndexf(i, j, k, 13)] - geq[13]) - (dev_g[getIndexf(i, j, k, 2)] - geq[2]) - (dev_g[getIndexf(i, j, k, 12)] - geq[12]) - (dev_g[getIndexf(i, j, k, 14)] - geq[14]);

//                 float Mz = (dev_g[getIndexf(i, j, k, 5)] - geq[5]) + (dev_g[getIndexf(i, j, k, 11)] - geq[11]) + (dev_g[getIndexf(i, j, k, 14)] - geq[14]) - (dev_g[getIndexf(i, j, k, 6)] - geq[6]) - (dev_g[getIndexf(i, j, k, 12)] - geq[12]) - (dev_g[getIndexf(i, j, k, 13)] - geq[13]);

//                 dev_g[getIndexf(i, j, k, 4)] = geq[4] + dev_g[getIndexf(i, j, k, 3)] - geq[3];
//                 dev_g[getIndexf(i, j, k, 8)] = dev_g[getIndexf(i, j, k, 7)] + geq[8] - geq[7] - (1.0 / 4.0) * (ex[8] * Mx + ez[8] * Mz);
//                 dev_g[getIndexf(i, j, k, 9)] = dev_g[getIndexf(i, j, k, 10)] + geq[9] - geq[10] - (1.0 / 4.0) * (ex[9] * Mx + ez[9] * Mz);
//                 dev_g[getIndexf(i, j, k, 16)] = dev_g[getIndexf(i, j, k, 15)] + geq[16] - geq[15] - (1.0 / 4.0) * (ex[16] * Mx + ez[16] * Mz);
//                 dev_g[getIndexf(i, j, k, 18)] = dev_g[getIndexf(i, j, k, 17)] + geq[18] - geq[17] - (1.0 / 4.0) * (ex[18] * Mx + ez[18] * Mz);

//                 float N = (dev_h[getIndexf(i, j, k, 1)] - heq[1]) + (dev_h[getIndexf(i, j, k, 2)] - heq[2]) + (dev_h[getIndexf(i, j, k, 5)] - heq[5]) + (dev_h[getIndexf(i, j, k, 6)] - heq[6]) + (dev_h[getIndexf(i, j, k, 11)] - heq[11]) +
//                           (dev_h[getIndexf(i, j, k, 12)] - heq[12]) + (dev_h[getIndexf(i, j, k, 13)] - heq[13]) +
//                           (dev_h[getIndexf(i, j, k, 14)] - heq[14]);

//                 dev_h[getIndexf(i, j, k, 4)] = heq[4] - (dev_h[getIndexf(i, j, k, 3)] - heq[3]);
//                 dev_h[getIndexf(i, j, k, 8)] = heq[8] - (dev_h[getIndexf(i, j, k, 7)] - heq[7]) - N / 4.0;
//                 dev_h[getIndexf(i, j, k, 9)] = heq[9] - (dev_h[getIndexf(i, j, k, 10)] - heq[10]) - N / 4.0;
//                 dev_h[getIndexf(i, j, k, 16)] = heq[16] - (dev_h[getIndexf(i, j, k, 15)] - heq[15]) - N / 4.0;
//                 dev_h[getIndexf(i, j, k, 18)] = heq[18] - (dev_h[getIndexf(i, j, k, 17)] - heq[17]) - N / 4.0;
//             }

//             else
//             {
//                 // dev_c[index] = 0.0L;
// #pragma unroll
//                 for (int q = 0; q < 5; q++)
//                 {
//                     dev_h[getIndexf(i, Ny_, k, FStop[q])] = dev_h[getIndexf(i, Ny_1, k, FStop[q + 5])];
//                     dev_g[getIndexf(i, Ny_, k, FStop[q])] = dev_g[getIndexf(i, Ny_1, k, FStop[q + 5])];
//                 }
//             }
        }
    }

    //////////////////////////////////////////////////////// end

    __global__ void kernelMirrorBoundary(const Domain::DomainInfo domain, const Boundary::BCTypeMap BCmap, CLIP_REAL *dev_a)
    {
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        if (Domain::isInside<DIM>(domain, i, j, k))
        {

            const CLIP_UINT idx = Domain::getIndex(domain, i, j, k);

            // XMIN
            if (Boundary::isMirrorType(BCmap.types[object::XMinus]) && i == domain.domainMinIdx[IDX_X])
            {
                const CLIP_UINT ghost = Domain::getIndex(domain, domain.ghostDomainMinIdx[IDX_X], j, k);
                dev_a[ghost] = dev_a[idx];
            }
            // XMAX
            if (Boundary::isMirrorType(BCmap.types[object::XPlus]) && i == domain.domainMaxIdx[IDX_X])
            {
                const CLIP_UINT ghost = Domain::getIndex(domain, domain.ghostDomainMaxIdx[IDX_X], j, k);
                dev_a[ghost] = dev_a[idx];
            }
            // YMIN
            if (Boundary::isMirrorType(BCmap.types[object::YMinus]) && j == domain.domainMinIdx[IDX_Y])
            {
                const CLIP_UINT ghost = Domain::getIndex(domain, i, domain.ghostDomainMinIdx[IDX_Y], k);
                dev_a[ghost] = dev_a[idx];
            }
            // YMAX
            if (Boundary::isMirrorType(BCmap.types[object::YPlus]) && j == domain.domainMaxIdx[IDX_Y])
            {
                const CLIP_UINT ghost = Domain::getIndex(domain, i, domain.ghostDomainMaxIdx[IDX_Y], k);
                dev_a[ghost] = dev_a[idx];
            }

#ifdef ENABLE_3D
            // ZMIN
            if (Boundary::isMirrorType(BCmap.types[object::ZMinus]) && k == domain.domainMinIdx[IDX_Z])
            {
                const CLIP_UINT ghost = Domain::getIndex(domain, i, j, domain.ghostDomainMinIdx[IDX_Z]);
                dev_a[ghost] = dev_a[idx];
            }
            // ZMAX
            if (Boundary::isMirrorType(BCmap.types[object::ZPlus]) && k == domain.domainMaxIdx[IDX_Z])
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
            kernelHalfBounceBack<Q, dof, WMRT::wallBCMap><<<dimGrid, dimBlock>>>(m_info, m_BCMap, m_wallBCMap, dev_a, dev_a_post, dev_b, dev_b_post);
    }

    template <CLIP_UINT Q, CLIP_UINT dof>
    void Solver::slipWallBoundary(CLIP_REAL *dev_a, CLIP_REAL *dev_a_post, CLIP_REAL *dev_b, CLIP_REAL *dev_b_post)
    {
        if (m_boundary->isSlipWall)
            kernelHalfBounceBack<Q, dof, WMRT::slipWallBCMap><<<dimGrid, dimBlock>>>(m_info, m_BCMap, m_slipWallBCMap, dev_a, dev_a_post, dev_b, dev_b_post);
    }

    template <CLIP_UINT Q, CLIP_UINT dof>
    void Solver::freeConvectBoundary(CLIP_REAL *dev_vel, CLIP_REAL *dev_a, CLIP_REAL *dev_a_prev, CLIP_REAL *dev_b, CLIP_REAL *dev_b_prev)
    {
        if (m_boundary->isFreeConvect)
            kernelFreeConvect<Q, dof, WMRT::wallBCMap><<<dimGrid, dimBlock>>>(m_info, m_BCMap, m_wallBCMap, dev_vel, dev_a, dev_a_prev, dev_b, dev_b_prev);
    }

    template <CLIP_UINT Q, CLIP_UINT dof>
    void Solver::NeumannBoundary(CLIP_REAL *dev_a, CLIP_REAL *dev_b)
    {
        if (m_boundary->isNeumann)
            kernelNeumann<Q, dof, WMRT::wallBCMap><<<dimGrid, dimBlock>>>(m_info, m_BCMap, m_wallBCMap, dev_a, dev_b);
    }

    void Solver::mirrorBoundary(CLIP_REAL *dev_a)
    {
        if (m_boundary->isWall || m_boundary->isFreeConvect || m_boundary->isSlipWall || m_boundary->isNeumann)
            kernelMirrorBoundary<<<dimGrid, dimBlock>>>(m_info, m_BCMap, dev_a);
    }




    void Solver::velocityBoundary(CLIP_REAL *dev_c, CLIP_REAL *dev_f, CLIP_REAL *dev_g)
    {
        // if (m_boundary->isNeumann)
            JetBoundary<<<dimGrid, dimBlock>>>(m_info, m_geomPool, m_BCMap, m_velSet, m_slipWallBCMap, dev_c, dev_f, dev_g);


    }





    template void clip::Solver::periodicBoundary<9>(CLIP_REAL *, CLIP_REAL *);
    template void clip::Solver::periodicBoundary<19>(CLIP_REAL *, CLIP_REAL *);
    template void clip::Solver::periodicBoundary<1>(CLIP_REAL *, CLIP_REAL *);

    template void clip::Solver::wallBoundary<9, 3>(CLIP_REAL *, CLIP_REAL *, CLIP_REAL *, CLIP_REAL *);
    template void clip::Solver::wallBoundary<19, 5>(CLIP_REAL *, CLIP_REAL *, CLIP_REAL *, CLIP_REAL *);

    template void clip::Solver::slipWallBoundary<9, 3>(CLIP_REAL *, CLIP_REAL *, CLIP_REAL *, CLIP_REAL *);
    template void clip::Solver::slipWallBoundary<19, 5>(CLIP_REAL *, CLIP_REAL *, CLIP_REAL *, CLIP_REAL *);

    template void clip::Solver::freeConvectBoundary<9, 3>(CLIP_REAL *, CLIP_REAL *, CLIP_REAL *, CLIP_REAL *, CLIP_REAL *);
    template void clip::Solver::freeConvectBoundary<19, 5>(CLIP_REAL *, CLIP_REAL *, CLIP_REAL *, CLIP_REAL *, CLIP_REAL *);

    template void clip::Solver::NeumannBoundary<9, 3>(CLIP_REAL *, CLIP_REAL *);
    template void clip::Solver::NeumannBoundary<19, 5>(CLIP_REAL *, CLIP_REAL *);

}
