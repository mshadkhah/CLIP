#include "DataArray.cuh"


namespace clip
{


    DataArray::DataArray(const InputData& idata, const Domain& domain, const Boundary& boundary)
    : m_idata(&idata), m_domain(&domain), m_boundary(&boundary)
{

    CLIP_INT threadsAlongX = 8, threadsAlongY = 8, threadsAlongZ = 1;
    dimBlock = dim3(threadsAlongX, threadsAlongY, threadsAlongZ);

#ifdef ENABLE_2D
    CLIP_INT gridX = static_cast<CLIP_INT>(std::ceil(CLIP_REAL(m_domain->info.extent[IDX_X]) / threadsAlongX));
    CLIP_INT gridY = static_cast<CLIP_INT>(std::ceil(CLIP_REAL(m_domain->info.extent[IDX_Y]) / threadsAlongY));
    dimGrid = dim3(gridX, gridY);

#elif defined(ENABLE_3D)

    CLIP_INT gridX = static_cast<CLIP_INT>(std::ceil(CLIP_REAL(m_domain->info.extent[IDX_X]) / threadsAlongX));
    CLIP_INT gridY = static_cast<CLIP_INT>(std::ceil(CLIP_REAL(m_domain->info.extent[IDX_Y]) / threadsAlongY));
    CLIP_INT gridZ = static_cast<CLIP_INT>(std::ceil(CLIP_REAL(m_domain->info.extent[IDX_Z]) / threadsAlongZ));
    dimGrid = dim3(gridX, gridY, gridZ);

#endif


};



void DataArray::createVectors()
{
    const CLIP_UINT Q = WMRT::WMRTvelSet::Q;

    if(m_boundary->isFreeConvect)
    {
            this->allocateOnDevice(deviceDA.dev_f_prev, "dev_f_prev", Q);
    this->allocateOnDevice(deviceDA.dev_g_prev, "dev_g_prev", Q);
    }


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

    this->allocateOnHost(hostDA.host_f, "host_f", Q);
    this->allocateOnHost(hostDA.host_g, "host_g", Q);
    this->allocateOnHost(hostDA.host_f_post, "host_f_post", Q);
    this->allocateOnHost(hostDA.host_g_post, "host_g_post", Q);
    if(m_boundary->isFreeConvect)
    {
    this->allocateOnHost(hostDA.host_f_prev, "host_f_prev", Q);
    this->allocateOnHost(hostDA.host_g_prev, "host_g_prev", Q);
    }

    Logger::Success("Host vectors are allocated successfully.");




}

void DataArray::updateDevice()
{
    const CLIP_UINT Q = WMRT::WMRTvelSet::Q;

    copyToDevice(deviceDA.dev_c, hostDA.host_c, "dev_c", SCALAR_FIELD);
    // copyToDevice(deviceDA.dev_vel, hostDA.host_vel, "dev_vel", DIM);

}


void DataArray::updateHost()
{
    const CLIP_UINT Q = WMRT::WMRTvelSet::Q;

    copyFromDevice(hostDA.host_c, deviceDA.dev_c, "host_c", SCALAR_FIELD);
    copyFromDevice(hostDA.host_p, deviceDA.dev_p, "host_p", SCALAR_FIELD);
    copyFromDevice(hostDA.host_rho, deviceDA.dev_rho, "host_rho", SCALAR_FIELD);
    copyFromDevice(hostDA.host_vel, deviceDA.dev_vel, "host_rho", DIM);

}









}