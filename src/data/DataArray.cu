#include <DataArray.cuh>

namespace clip
{

    DataArray::DataArray(InputData idata)
        : m_idata(idata)
    {
        m_nVelocity = m_idata.nVelocity;

#ifdef ENABLE_2D
        domainDimension = (m_idata.Nx + 2) * (m_idata.Ny + 2);
#elif defined(ENABLE_3D)
        domainDimension = (m_idata.Nx + 2) * (m_idata.Ny + 2) * (m_idata.Nz + 2);
#endif
        latticeDimension = domainDimension * m_nVelocity;
    };
}