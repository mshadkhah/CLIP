#include <NsAllen.cuh>
#include <Solver.cuh>
#include <NSAllenKernels.cu>


namespace clip
{

    NSAllen::NSAllen(InputData idata)
        : m_idata(idata), Solver(idata)
    {

        flagGenLauncher2(m_velset);

        // initialization();
    };

    // void NSAllen::setVectors()
    // {
    //     const CLIP_UINT Q = m_velset.Q;

    //     this->allocateOnDevice(dev_f, "dev_f", Q);
    //     this->allocateOnDevice(dev_g, "dev_g", Q);
    //     this->allocateOnDevice(dev_f_post, "dev_f_post", Q);
    //     this->allocateOnDevice(dev_g_post, "dev_g_post", Q);

    //     this->allocateOnDevice(dev_rho, "dev_rho", SCALAR_FIELD);
    //     this->allocateOnDevice(dev_mu, "dev_mu", SCALAR_FIELD);
    //     this->allocateOnDevice(dev_c, "dev_c", SCALAR_FIELD);
    //     this->allocateOnDevice(dev_p, "dev_p", SCALAR_FIELD);

    //     this->allocateOnDevice(dev_vel, "dev_vel", DIM);
    //     this->allocateOnDevice(dev_dc, "dev_dc", DIM);
    //     this->allocateOnDevice(dev_normal, "dev_normal", DIM);

    //     this->allocateOnHost(host_c, "host_c", SCALAR_FIELD);
    // }

    // void NSAllen::solve()
    // {

    //     flagGen2<<<dimGrid, dimBlock>>>(velSet);
    //     cudaDeviceSynchronize();

    // }

    NSAllen::~NSAllen()
    {

        SAFE_CUDA_FREE(dev_f);
        SAFE_CUDA_FREE(dev_f_post);
        SAFE_CUDA_FREE(dev_g);
        SAFE_CUDA_FREE(dev_g_post);
    }

    __global__ void flagGen2(const Solver::WMRTvelSet velSet)
    {
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        constexpr CLIP_UINT Q = Solver::WMRTvelSet::Q;

        // const CLIP_UINT idx_SCALAR = DataArray::getIndex(i, j, k);

        // printf("Thread index: i = %d, j = %d, k = %d\n", i, j, k);
        // printf("Thread index2: %d \n", idx_SCALAR);

        // if (DataArray::isInside<DIM>(i, j, k)){

        // printf("Thread index: i = %d, j = %d, k = %d\n", i, j, k);
        // printf("index: i = %d\n", DataArray::getDomainExtent(IDX_X));
        // printf("index: s_domainExtent = %d\n", s_domainExtent[IDX_Y]);

        // }
        // printf("index: inside allen s_domainExtent = %d\n", DataArray::getDomainExtent(1));
        // printf("index: i = %d\n", idx_SCALAR);
        // printf("Thread index: i = %d, j = %d, k = %d\n", i, j, k);

        printf("index:  inside allen ex = %d\n", Q);
    }

    void NSAllen::flagGenLauncher2(const Solver::WMRTvelSet velSet)
    {

        flagGen2<<<dimGrid, dimBlock>>>(velSet);
        cudaDeviceSynchronize();
    }

    __device__ __forceinline__ CLIP_REAL NSAllen::Equilibrium_new(const Solver::WMRTvelSet velSet, int q, CLIP_REAL Ux, CLIP_REAL Uy, CLIP_REAL Uz = 0)
    {

        const CLIP_REAL exq = velSet.ex[q];
        const CLIP_REAL eyq = velSet.ey[q];
        const CLIP_REAL waq = velSet.wa[q];

#ifdef ENABLE_2D
        const CLIP_REAL eU = exq * Ux + eyq * Uy;
        const CLIP_REAL U2 = Ux * Ux + Uy * Uy;
#elif defined(ENABLE_3D)
        const CLIP_REAL ezq = velSet.ez[q];
        const CLIP_REAL eU = exq * Ux + eyq * Uy + ezq * Uz;
        const CLIP_REAL U2 = Ux * Ux + Uy * Uy + Uz * Uz;
#endif

        return waq * (3.0 * eU + 4.5 * eU * eU - 1.5 * U2);
    }

    template <CLIP_UINT q, size_t dim>
    __device__ __forceinline__ void NSAllen::calculateVF(const Solver::WMRTvelSet velSet, const InputData::SimParams params, CLIP_REAL gneq[q], CLIP_REAL fv[dim], CLIP_REAL tau, CLIP_REAL dcdx, CLIP_REAL dcdy, CLIP_REAL dcdz)
    {
        CLIP_REAL sxx = 0;
        CLIP_REAL sxy = 0;
        CLIP_REAL syy = 0;
        CLIP_REAL szy = 0;
        CLIP_REAL szx = 0;
#ifdef ENABLE_3D
        CLIP_REAL szz = 0;
#endif
        const CLIP_REAL rhoDiff = params.RhoH - params.RhoL;

#pragma unroll
        for (int i = 0; i < q; i++)
        {
            sxx += velSet.ex[i] * velSet.ex[i] * gneq[i];
            sxy += velSet.ex[i] * velSet.ey[i] * gneq[i];
            syy += velSet.ey[i] * velSet.ey[i] * gneq[i];
#ifdef ENABLE_3D
            szy += velSet.ez[i] * velSet.ey[i] * gneq[i];
            szx += velSet.ez[i] * velSet.ex[i] * gneq[i];
            szz += velSet.ez[i] * velSet.ez[i] * gneq[i];
#endif
        }
        fv[IDX_X] = -tau * (sxx * dcdx + sxy * dcdy + szx * dcdz) * rhoDiff;
        fv[IDX_Y] = -tau * (sxy * dcdx + syy * dcdy + szy * dcdz) * rhoDiff;
#ifdef ENABLE_3D
        fv[IDX_Z] = -tau * (szz * dcdz + szx * dcdx + szy * dcdy) * rhoDiff;
#endif
    }


    void NSAllen::collision()
    {

        normal_FD<<<dimGrid, dimBlock>>>(m_domain.info, dev_dc, dev_normal);

        kernelCollisionMRTh<<<dimGrid, dimBlock>>>(m_velset, m_idata.params, m_domain.info,
                                                   dev_g, dev_g_post, dev_c, dev_rho, dev_vel, dev_normal);

        kernelCollisionMRTg<<<dimGrid, dimBlock>>>(m_velset, m_idata.params, m_domain.info,
                                                   dev_f, dev_f_post, dev_p, dev_c, dev_dc, dev_mu, dev_rho, dev_vel, dev_normal);

        cudaDeviceSynchronize();
    }

    void NSAllen::streaming()
    {

        kernelStreaming<<<dimGrid, dimBlock>>>(m_velset, m_domain.info, dev_f, dev_f_post);
        kernelStreaming<<<dimGrid, dimBlock>>>(m_velset, m_domain.info, dev_g, dev_g_post);

        cudaDeviceSynchronize();
    }

    void NSAllen::macroscopic()
    {
        kernelMacroscopicg<<<dimGrid, dimBlock>>>(m_velset, m_idata.params, m_domain.info,
                                                  dev_p, dev_g_post, dev_rho, dev_c);

        Chemical_Potential<<<dimGrid, dimBlock>>>(m_idata.params, m_domain.info, dev_c, dev_mu);
        Isotropic_Gradient<<<dimGrid, dimBlock>>>(m_domain.info, dev_c, dev_dc);
        kernelMacroscopicf<<<dimGrid, dimBlock>>>(m_velset, m_idata.params, m_domain.info,
                                                  dev_p, dev_rho, dev_c, dev_f_post, dev_dc, dev_vel, dev_mu);
    }

    void NSAllen::solve()
    {
        constexpr CLIP_UINT Q = Solver::WMRTvelSet::Q;

        collision();
        periodicBoundary<Q>(dev_f, dev_g);
        periodicBoundary<SCALAR_FIELD>(dev_c);
        streaming();
    }





    void NSAllen::initializer()
    {

        for (CLIP_UINT i = m_domain.info.ghostDomainMinIdx[IDX_X]; i <= m_domain.info.ghostDomainMaxIdx[IDX_X]; i++)
        {
            for (CLIP_UINT j = m_domain.info.ghostDomainMinIdx[IDX_Y]; j <= m_domain.info.ghostDomainMaxIdx[IDX_Y]; j++)
            {
                for (CLIP_UINT k = m_domain.info.ghostDomainMinIdx[IDX_Z]; k <= m_domain.info.ghostDomainMaxIdx[IDX_Z]; k++)
                {

                    const CLIP_UINT idx_SCALAR = Domain::getIndex(m_domain.info, i, j, k);

                    printf("index:  inside initi index = %d\n", idx_SCALAR);

                    const CLIP_REAL X0 = i - m_idata.params.C[IDX_X];
                    const CLIP_REAL Y0 = j - m_idata.params.C[IDX_Y];
                    const CLIP_REAL Z0 = k - m_idata.params.C[IDX_Z];

                    std::cout << "i = " << i << ", j = " << j << ", k = " << k << std::endl;

                    const CLIP_REAL Ri = sqrt(X0 * X0 + Y0 * Y0 + Z0 * Z0);

                    if (m_idata.params.caseType == InputData::CaseType::Bubble)
                    {
                        host_c[idx_SCALAR] = 0.50L - 0.50L * tanh(2.0L * (m_idata.params.radius - Ri) / m_idata.params.interfaceWidth);
                    }
                    else if (m_idata.params.caseType == InputData::CaseType::Drop)
                    {
                        host_c[idx_SCALAR] = 0.50L + 0.50L * tanh(2.0L * (m_idata.params.radius - Ri) / m_idata.params.interfaceWidth);
                    }
                    else
                    {
                        host_c[idx_SCALAR] = 0.0; // fallback
                    }
                }
            }
        }

        this->copyToDevice(dev_c, host_c, "dev_c", SCALAR_FIELD);
        // Chemical_Potential<<<dimGrid, dimBlock>>>(m_idata.params, m_domain.info, dev_c, dev_mu);

        // Isotropic_Gradient<<<dimGrid, dimBlock>>>(m_domain.info, dev_c, dev_dc);

        // normal_FD<<<dimGrid, dimBlock>>>(m_domain.info, dev_dc, dev_normal);

        // KernelInitializeDistributions<<<dimGrid, dimBlock>>>(m_velset, m_idata.params, m_domain.info, dev_f, dev_g, dev_f_post, dev_g_post,
        //     dev_c, dev_rho, dev_p, dev_vel, dev_normal);

            
        // cudaDeviceSynchronize();

        // cudaMemcpy(dev_c, &host_c[0], dim * sizeof(double), cudaMemcpyHostToDevice);
        // cudaCheckErrors("cudaMemcpyHostToDevice 'dev_c' fail");

        // Chemical_Potential<<<dimGrid, dimBlock>>>(dev_c, dev_mu, dev_beta, dev_kk);
        // Isotropic_Gradient<<<dimGrid, dimBlock>>>(dev_c, dev_dcdx, dev_dcdy, dev_dcdz);
        // normal_FD<<<dimGrid, dimBlock>>>(dev_dcdx, dev_dcdy, dev_dcdz, dev_ni, dev_nj, dev_nk);
        // KernelInitializeDistributions<<<dimGrid, dimBlock>>>(dev_h, dev_g, dev_h_post, dev_g_post, dev_c, dev_p, dev_ux, dev_uy, dev_uz, dev_rho, dev_ni, dev_nj, dev_nk, dev_rhol, dev_rhoh, dev_sigma, dev_r0, dev_w, dev_u0, dev_x0, dev_y0, dev_z0);
    }










}