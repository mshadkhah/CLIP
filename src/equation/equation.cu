#include <equation.cuh>







namespace clip {

    Equation::Equation(InputData idata)
    : m_idata(idata), DataArray(idata){




        m_nVelocity = m_idata.nVelocity;

        const CLIP_UINT XMinus = 1;
        const CLIP_UINT XMinus_G = 0;
        const CLIP_UINT YMinus = 1;
        const CLIP_UINT YMinus_G = 0;
        const CLIP_UINT ZMinus = 1;
        const CLIP_UINT ZMinus_G = 0;

        const CLIP_UINT XPlus = m_idata.N[IDX_X] - 1;
        const CLIP_UINT XPlus_G = m_idata.N[IDX_X];
        const CLIP_UINT YPlus = m_idata.N[IDX_Y] - 1;
        const CLIP_UINT YPlus_G = m_idata.N[IDX_Y];
        const CLIP_UINT ZPlus = m_idata.N[IDX_Z] - 1;
        const CLIP_UINT ZPlus_G = m_idata.N[IDX_Z];


        this->symbolOnDevice(boundary::s_XMinus, &XMinus, "s_XMinus");
        this->symbolOnDevice(boundary::s_XMinus_G, &XMinus_G, "s_XMinus_G");
        this->symbolOnDevice(boundary::s_YMinus, &YMinus, "s_XMinus");
        this->symbolOnDevice(boundary::s_YMinus_G, &YMinus_G, "s_XMinus_G");
        this->symbolOnDevice(boundary::s_ZMinus, &ZMinus, "s_ZMinus");
        this->symbolOnDevice(boundary::s_ZMinus_G, &ZMinus_G, "s_ZMinus_G");

        this->symbolOnDevice(boundary::s_XPlus, &XPlus, "s_XPlus");
        this->symbolOnDevice(boundary::s_XPlus_G, &XPlus_G, "s_XPlus_G");
        this->symbolOnDevice(boundary::s_YPlus, &YPlus, "s_YPlus");
        this->symbolOnDevice(boundary::s_YPlus_G, &YPlus_G, "s_YPlus_G");
        this->symbolOnDevice(boundary::s_ZPlus, &ZPlus, "s_ZPlus");
        this->symbolOnDevice(boundary::s_ZPlus_G, &ZPlus_G, "s_ZPlus_G");



#ifdef ENABLE_2D
        m_ex = new CLIP_INT[WMRT::Q]{0, 1, 0, -1, 0, 1, -1, -1, 1};
        m_ey = new CLIP_INT[WMRT::Q]{0, 0, 1, 0, -1, 1, 1, -1, -1};
        m_wa = new CLIP_REAL[WMRT::Q]{4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

        this->symbolOnDevice(WMRT::ex, m_ex, "ex");
        this->symbolOnDevice(WMRT::ey, m_ey, "ey");
        this->symbolOnDevice(WMRT::wa, m_wa, "wa");

#elif defined(ENABLE_3D)
        m_ex = new CLIP_INT[WMRT::Q]{0, 1, 0, -1, 0, 1, -1, -1, 1};
        m_ey = new CLIP_INT[WMRT::Q]{0, 0, 1, 0, -1, 1, 1, -1, -1};
        m_ez = new CLIP_INT[WMRT::Q]{0, 0, 1, 0, -1, 1, 1, -1, -1};
        m_wa = new CLIP_REAL[WMRT::Q]{4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

        this->symbolOnDevice(WMRT::ex, m_ex, "ex");
        this->symbolOnDevice(WMRT::ey, m_ey, "ey");
        this->symbolOnDevice(WMRT::ez, m_ez, "ez");
        this->symbolOnDevice(WMRT::wa, m_wa, "wa");

#endif



this->symbolOnDevice(boundary::s_boundaries, m_idata.boundaries.data(), "boundaries");




    }



    Equation::~Equation() {

    if (m_ex)
        delete[] m_ex;
    if (m_ey)
        delete[] m_ey;
    if (m_ez)
        delete[] m_ez;
    if (m_wa)
        delete[] m_wa;

    }



    template <int dof = 1>
    __global__ void periodicBoundary(double* dev_a, double* dev_b = nullptr)
    {
        using namespace boundary;

        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = DataArray::getIndex(i, j, k);
        const CLIP_UINT idx_X = DataArray::getIndex<DIM>(i, j, k, IDX_X);
        const CLIP_UINT idx_Y = DataArray::getIndex<DIM>(i, j, k, IDX_Y);

#ifdef ENABLE_3D
        const CLIP_UINT idx_Z = DataArray::getIndex<DIM>(i, j, k, IDX_Z);
#endif

        if (DataArray::isInside<DIM>(i, j, k))
        {

            #pragma unroll
            for (int q = 0; q < dof; ++q)
            {

            if(s_boundaries[clip::InputBoundary::Side::XMinus].flagCheck(clip::InputBoundary::Type::Periodic)){

                dev_a[DataArray::getIndex<dof>(i, j, k, q)] = dev_a[DataArray::getIndex<dof>(i, j, k, q)];
                if(dev_b)
                dev_b[DataArray::getIndex<dof>(i, j, k, q)] = dev_b[DataArray::getIndex<dof>(i, j, k, q)];
            }

            if(s_boundaries[clip::InputBoundary::Side::YMinus].flagCheck(clip::InputBoundary::Type::Periodic)){
                dev_a[DataArray::getIndex<dof>(i, j, k, q)] = dev_a[DataArray::getIndex<dof>(i, j, k, q)];
                if(dev_b)
                dev_b[DataArray::getIndex<dof>(i, j, k, q)] = dev_b[DataArray::getIndex<dof>(i, j, k, q)];
            }

#ifdef ENABLE_3D
if(s_boundaries[clip::InputBoundary::Side::ZMinus].flagCheck(clip::InputBoundary::Type::Periodic)){
    dev_a[DataArray::getIndex<dof>(i, j, k, q)] = dev_a[DataArray::getIndex<dof>(i, j, k, q)];
    if(dev_b)
    dev_b[DataArray::getIndex<dof>(i, j, k, q)] = dev_b[DataArray::getIndex<dof>(i, j, k, q)];
}

#endif
    

        }













        {
    #pragma unroll
            for (int q = 0; q < dim; ++q)
            {
                // Example: uncomment or modify as needed
                // dev_h[getIndexf(Nx_, j, q)] = dev_h[getIndexf(1, j, q)];
                // dev_g[getIndexf(Nx_, j, q)] = dev_g[getIndexf(1, j, q)];
    
                // dev_h[getIndexf(0, j, q)] = dev_h[getIndexf(Nx_1, j, q)];
                // dev_g[getIndexf(0, j, q)] = dev_g[getIndexf(Nx_1, j, q)];
            }
        }
    }
    
    template <int Q>
    void Equation::launchPeriodicBoundaryF(
        double* dev_h,
        double* dev_g,
        CLIP_UINT Nx,
        CLIP_UINT Ny,
        CLIP_UINT Nz  // Optional for 2D
    ) {
        dim3 blockDim(16, 16, 1);  // You can adjust for optimal performance
    
    #ifdef ENABLE_3D
        dim3 gridDim(
            (Nx + blockDim.x - 1) / blockDim.x,
            (Ny + blockDim.y - 1) / blockDim.y,
            (Nz + blockDim.z - 1) / blockDim.z);
    #else
        dim3 gridDim(
            (Nx + blockDim.x - 1) / blockDim.x,
            (Ny + blockDim.y - 1) / blockDim.y);
    #endif
    
        // Call the templated kernel
        periodicBoundary<Q><<<gridDim, blockDim>>>(dev_h, dev_g);
    
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        }
    
        cudaDeviceSynchronize();  // Optional
    }
    




}

