#include <DataArray.cuh>

namespace clip
{

    DataArray::DataArray(InputData idata)
        : m_idata(idata)
    {
        m_nVelocity = m_idata.nVelocity;

#ifdef ENABLE_2D
        m_domainExtent = new CLIP_UINT[DIM]{m_idata.Nx + 2, m_idata.Ny + 2};
        symbolOnDevice(domainExtent, m_domainExtent, "wdomainExtenta");
        // m_domainExtentGhosted = new CLIP_UINT[DIM]{m_idata.Nx + 2, m_idata.Ny + 2};
        // symbolOnDevice(domainExtentGhosted, m_domainExtentGhosted, "wdomainExtenta");

        domainSize = (m_idata.Nx + 2) * (m_idata.Ny + 2);
#elif defined(ENABLE_3D)

        m_domainExtent = new CLIP_UINT[DIM]{m_idata.Nx + 2, m_idata.Ny + 2, m_idata.Nz + 2};
        symbolOnDevice(domainExtent, m_domainExtent, "wdomainExtenta");
        // m_domainExtentGhosted = new CLIP_UINT[DIM]{m_idata.Nx + 2, m_idata.Ny + 2, m_idata.Nz + 2};
        // symbolOnDevice(domainExtentGhosted, m_domainExtentGhosted, "wdomainExtenta");

        domainSize = (m_idata.Nx + 2) * (m_idata.Ny + 2) * (m_idata.Nz + 2);
#endif
        latticeSize = domainSize * m_nVelocity;
    };
}