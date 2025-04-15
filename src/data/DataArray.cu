#include <DataArray.cuh>



namespace clip
{

    DataArray::DataArray(InputData idata)
        : m_idata(idata)
    {
        m_nVelocity = m_idata.nVelocity;



    CLIP_INT threadsAlongX = 8, threadsAlongY = 8, threadsAlongZ = 1;
    dimBlock = dim3(threadsAlongX, threadsAlongY, threadsAlongZ);


#ifdef ENABLE_2D
        m_domainExtent = new CLIP_UINT[MAX_DIM]{m_idata.N[IDX_X] + 2, m_idata.N[IDX_Y] + 2, 1};
        symbolOnDevice(s_domainExtent, m_domainExtent, "domainExtent");
        domainSize = (m_idata.N[IDX_X] + 2) * (m_idata.N[IDX_Y] + 2);

        CLIP_INT gridX = static_cast<CLIP_INT>(std::ceil(m_domainExtent[IDX_X] / threadsAlongX));
        CLIP_INT gridY = static_cast<CLIP_INT>(std::ceil(m_domainExtent[IDX_Y] / threadsAlongY));
        dimGrid = dim3(gridX, gridY);


        int host_flag[MAX_DIM];
        cudaMemcpyFromSymbol(&host_flag, s_domainExtent, MAX_DIM * sizeof(CLIP_UINT));

#elif defined(ENABLE_3D)

        m_domainExtent = new CLIP_UINT[DIM]{m_idata.N[IDX_X] + 2, m_idata.N[IDX_Y] + 2, m_idata.N[IDX_Z] + 2};
        symbolOnDevice(s_domainExtent, m_domainExtent, "domainExtent");
        domainSize = (m_idata.N[IDX_X] + 2) * (m_idata.N[IDX_Y] + 2) * (m_idata.N[IDX_Z] + 2);

    CLIP_INT gridX = static_cast<CLIP_INT>(std::ceil(m_domainExtent[IDX_X] / threadsAlongX));
    CLIP_INT gridY = static_cast<CLIP_INT>(std::ceil(m_domainExtent[IDX_Y] / threadsAlongY));
    CLIP_INT gridZ = static_cast<CLIP_INT>(std::ceil(m_domainExtent[IDX_Z] / threadsAlongZ));
    dimGrid = dim3(gridX, gridY, gridZ);


#endif
        latticeSize = domainSize * m_nVelocity;
    };




    






}