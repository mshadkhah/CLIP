#include <Domain.cuh>

namespace clip
{

    Domain::Domain(const InputData& idata)
        : m_idata(&idata)
    {

#ifdef ENABLE_2D



        info.extent[IDX_X] = m_idata->params.N[IDX_X] + 2;
        info.extent[IDX_Y] = m_idata->params.N[IDX_Y] + 2;
        info.extent[IDX_Z] = 1;

        info.ghostDomainMinIdx[IDX_X] = 0;
        info.ghostDomainMinIdx[IDX_Y] = 0;
        info.ghostDomainMinIdx[IDX_Z] = 0;

        info.ghostDomainMaxIdx[IDX_X] = info.extent[IDX_X] - 1;
        info.ghostDomainMaxIdx[IDX_Y] = info.extent[IDX_Y] - 1;
        info.ghostDomainMaxIdx[IDX_Z] = 0;



        info.domainMinIdx[IDX_X] = 1;
        info.domainMinIdx[IDX_Y] = 1;
        info.domainMinIdx[IDX_Z] = 0;


        info.domainMaxIdx[IDX_X] = info.extent[IDX_X] - 2;
        info.domainMaxIdx[IDX_Y] = info.extent[IDX_Y] - 2;
        info.domainMaxIdx[IDX_Z] = 0;

        domainSize = info.extent[IDX_X] * info.extent[IDX_Y];

#elif defined(ENABLE_3D)

        info.extent[IDX_X] = m_idata->params.N[IDX_X];
        info.extent[IDX_Y] = m_idata->params.N[IDX_Y];
        info.extent[IDX_Z] = m_idata->params.N[IDX_Z];

        info.domainMinIdx[IDX_X] = 1;
        info.domainMinIdx[IDX_Y] = 1;
        info.domainMinIdx[IDX_Z] = 1;

        info.domainMaxIdx[IDX_X] = info.extent[IDX_X] - 2;
        info.domainMaxIdx[IDX_Y] = info.extent[IDX_Y] - 2;
        info.domainMaxIdx[IDX_Z] = info.extent[IDX_Z] - 2;


        info.ghostDomainMinIdx[IDX_X] = 0;
        info.ghostDomainMinIdx[IDX_Y] = 0;
        info.ghostDomainMinIdx[IDX_Z] = 0;

        info.ghostDomainMaxIdx[IDX_X] = info.extent[IDX_X] - 1;
        info.ghostDomainMaxIdx[IDX_Y] = info.extent[IDX_Y] - 1;
        info.ghostDomainMaxIdx[IDX_Z] = info.extent[IDX_Z] - 1;

        domainSize = info.extent[IDX_X] * info.extent[IDX_Y] * info.extent[IDX_Z];
#endif
    };

}