#include "DataArray.cuh"


namespace clip
{


    DataArray::DataArray(const InputData& idata, const Domain& domain)
    : m_idata(&idata), m_domain(&domain)
{

    CLIP_INT threadsAlongX = 8, threadsAlongY = 8, threadsAlongZ = 1;
    dimBlock = dim3(threadsAlongX, threadsAlongY, threadsAlongZ);

#ifdef ENABLE_2D
    CLIP_INT gridX = static_cast<CLIP_INT>(std::ceil(m_domain->info.extent[IDX_X] / threadsAlongX));
    CLIP_INT gridY = static_cast<CLIP_INT>(std::ceil(m_domain->info.extent[IDX_Y] / threadsAlongY));
    dimGrid = dim3(gridX, gridY);

#elif defined(ENABLE_3D)

    CLIP_INT gridX = static_cast<CLIP_INT>(std::ceil(m_domain->info.extent[IDX_X] / threadsAlongX));
    CLIP_INT gridY = static_cast<CLIP_INT>(std::ceil(m_domain->info.extent[IDX_Y] / threadsAlongY));
    CLIP_INT gridZ = static_cast<CLIP_INT>(std::ceil(m_domain->info.extent[IDX_Z] / threadsAlongZ));
    dimGrid = dim3(gridX, gridY, gridZ);

#endif


};



void DataArray::createVectors()
{
    const CLIP_UINT Q = WMRT::WMRTvelSet::Q;

    this->allocateOnDevice(deviceDA.dev_f, "dev_f", Q);
    this->allocateOnDevice(deviceDA.dev_g, "dev_g", Q);
    this->allocateOnDevice(deviceDA.dev_f_post, "dev_f_post", Q);
    this->allocateOnDevice(deviceDA.dev_g_post, "dev_g_post", Q);

    this->allocateOnDevice(deviceDA.dev_rho, "dev_rho", SCALAR_FIELD);
    this->allocateOnDevice(deviceDA.dev_mu, "dev_mu", SCALAR_FIELD);
    this->allocateOnDevice(deviceDA.dev_c, "dev_c", SCALAR_FIELD);
    this->allocateOnDevice(deviceDA.dev_p, "dev_p", SCALAR_FIELD);

    this->allocateOnDevice(deviceDA.dev_vel, "dev_vel", DIM);
    this->allocateOnDevice(deviceDA.dev_dc, "dev_dc", DIM);
    this->allocateOnDevice(deviceDA.dev_normal, "dev_normal", DIM);
    
    Logger::Success("Device vectors are allocated successfully.");

    this->allocateOnHost(hostDA.host_c, "host_c", SCALAR_FIELD);
    this->allocateOnHost(hostDA.host_p, "host_p", SCALAR_FIELD);
    this->allocateOnHost(hostDA.host_vel, "host_vel", DIM);
    this->allocateOnHost(hostDA.host_rho, "host_rho", SCALAR_FIELD);

    Logger::Success("Host vectors are allocated successfully.");




}

void DataArray::updateDevice()
{
    const CLIP_UINT Q = WMRT::WMRTvelSet::Q;

    copyToDevice(deviceDA.dev_c, hostDA.host_c, "dev_c", SCALAR_FIELD);
    copyToDevice(deviceDA.dev_vel, hostDA.host_vel, "dev_vel", DIM);

    // allocateOnDevice(deviceDA.dev_f, "dev_f", Q);
    // allocateOnDevice(deviceDA.dev_g, "dev_g", Q);
    // allocateOnDevice(deviceDA.dev_f_post, "dev_f_post", Q);
    // allocateOnDevice(deviceDA.dev_g_post, "dev_g_post"
    // allocateOnDevice(deviceDA.dev_rho, "dev_rho", SCALAR_FIELD);
    // allocateOnDevice(deviceDA.dev_mu, "dev_mu", SCALAR_FIELD);
    // allocateOnDevice(deviceDA.dev_c, "dev_c", SCALAR_FIELD);
    // allocateOnDevice(deviceDA.dev_p, "dev_p", SCALAR_FI
    // allocateOnDevice(deviceDA.dev_vel, "dev_vel", DIM);
    // allocateOnDevice(deviceDA.dev_dc, "dev_dc", DIM);
    // allocateOnDevice(deviceDA.dev_normal, "dev_normal", DIM);
    
    // Logger::Success("Device vectors are allocated successfully.");

    // allocateOnHost(hostDA.host_c, "host_c", SCALAR_FIELD);
    // allocateOnHost(hostDA.host_p, "host_p", SCALAR_FIELD);
    // allocateOnHost(hostDA.host_vel, "host_vel", DIM);
    // allocateOnHost(hostDA.host_rho, "host_rho", SCALAR_FIELD);

    // Logger::Success("Host vectors are allocated successfully.");




}


void DataArray::updateHost()
{
    const CLIP_UINT Q = WMRT::WMRTvelSet::Q;

    copyFromDevice(hostDA.host_c, deviceDA.dev_c, "host_c", SCALAR_FIELD);

}





//     DataArray::DataArray(InputData idata)
//     : m_idata(idata)
// {
//     m_nVelocity = m_idata.nVelocity;

//     CLIP_INT threadsAlongX = 8, threadsAlongY = 8, threadsAlongZ = 1;
//     dimBlock = dim3(threadsAlongX, threadsAlongY, threadsAlongZ);

// #ifdef ENABLE_2D
//     m_domainExtent = new CLIP_UINT[MAX_DIM]{m_idata.N[IDX_X] + 2, m_idata.N[IDX_Y] + 2, 1};
//     symbolOnDevice(s_domainExtent, m_domainExtent, "domainExtent");
//     domainSize = (m_idata.N[IDX_X] + 2) * (m_idata.N[IDX_Y] + 2);

//     CLIP_INT gridX = static_cast<CLIP_INT>(std::ceil(m_domainExtent[IDX_X] / threadsAlongX));
//     CLIP_INT gridY = static_cast<CLIP_INT>(std::ceil(m_domainExtent[IDX_Y] / threadsAlongY));
//     dimGrid = dim3(gridX, gridY);

//     int host_flag[MAX_DIM];
//     cudaMemcpyFromSymbol(&host_flag, s_domainExtent, MAX_DIM * sizeof(CLIP_UINT));

// #elif defined(ENABLE_3D)

//     m_domainExtent = new CLIP_UINT[DIM]{m_idata.N[IDX_X] + 2, m_idata.N[IDX_Y] + 2, m_idata.N[IDX_Z] + 2};
//     symbolOnDevice(s_domainExtent, m_domainExtent, "domainExtent");
//     domainSize = (m_idata.N[IDX_X] + 2) * (m_idata.N[IDX_Y] + 2) * (m_idata.N[IDX_Z] + 2);

//     CLIP_INT gridX = static_cast<CLIP_INT>(std::ceil(m_domainExtent[IDX_X] / threadsAlongX));
//     CLIP_INT gridY = static_cast<CLIP_INT>(std::ceil(m_domainExtent[IDX_Y] / threadsAlongY));
//     CLIP_INT gridZ = static_cast<CLIP_INT>(std::ceil(m_domainExtent[IDX_Z] / threadsAlongZ));
//     dimGrid = dim3(gridX, gridY, gridZ);

// #endif
//     latticeSize = domainSize * m_nVelocity;


//     domain[1] = 15;


// };










}