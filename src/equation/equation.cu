
#include <equation.cuh>


namespace clip {

    Equation::Equation(InputData idata)
    : m_idata(idata), DataArray(idata){

        m_nVelocity = m_idata.nVelocity;

    }



    Equation::~Equation() {
    }







    template <int Q>
    __global__ void periodicBoundary(double* dev_h, double* dev_g)
    {
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;
    
        // Example placeholder for index bounds check (optional):
        // if (i < N[0] && j < N[1])
        {
    #pragma unroll
            for (int q = 0; q < Q; ++q)
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

