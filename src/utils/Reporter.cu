#include "Reporter.cuh"

namespace clip
{

    Reporter::Reporter(DataArray &DA, const InputData &idata, const Domain &domain, const TimeInfo &ti)
        : m_DA(&DA), m_idata(&idata), m_domain(&domain), m_ti(&ti)
    {
    }


    __global__ void sumReductionKernel(const CLIP_REAL *__restrict__ data, CLIP_REAL *partialSums,
                                      Domain::DomainInfo domain)
    {
        extern __shared__ double sdata[];
        int tid = threadIdx.x;

        int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

        int Nx = domain.extent[IDX_X];
        int Ny = domain.extent[IDX_Y];
        int Nz = domain.extent[IDX_Z];

        int i = globalIdx % Nx;
        int j = (globalIdx / Nx) % Ny;
        int k = globalIdx / (Nx * Ny);

        double value = 0.0;
        if (Domain::isInside<DIM, true>(domain, i, j, k))
        {
            int idx = Domain::getIndex(domain, i, j, k);
            value = data[idx];
        }

        sdata[tid] = value;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }

        if (tid == 0)
            partialSums[blockIdx.x] = sdata[0];
    }



    CLIP_REAL Reporter::sum(CLIP_REAL* dev_data, CLIP_UINT dof)
    {
        const int threadsPerBlock = 256;
        const int blocks = (dof + threadsPerBlock - 1) / threadsPerBlock;
    
        double* dev_partialSums;
        cudaMalloc(&dev_partialSums, blocks * sizeof(double));
    
        sumReductionKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(CLIP_REAL)>>>(dev_data, dev_partialSums, m_domain->info);

        std::vector<double> h_partialSums(blocks);
        cudaMemcpy(h_partialSums.data(), dev_partialSums, blocks * sizeof(CLIP_REAL), cudaMemcpyDeviceToHost);
        cudaFree(dev_partialSums);
    
        double totalSum = 0.0;
        for (int i = 0; i < blocks; ++i)
            totalSum += h_partialSums[i];
    
        return totalSum;
    }


    void Reporter::print()
    {
        const CLIP_UINT step = m_ti->getCurrentStep();
    
        if (step % m_idata->params.reportInterval == 0)
        {
            CLIP_REAL sumC = sum(m_DA->deviceDA.dev_c, m_domain->domainSize);


            std::ostringstream oss;
            oss << "\n===================== Solver Status =====================\n"
                << "Step        : " << step << "\n"
                // << "Time        : " << time << "\n"
                << "Sum(c)      : " << sumC << "\n"
                << "Domain Size : " << m_domain->domainSize << "\n"
                << "---------------------------------------------------------";
            
            Logger::Info(oss.str());
            if (std::isnan(sumC)) 
                Logger::Error("Solver has diverged.");

        }
    }
    
    




}