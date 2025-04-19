#include <NsAllen.cuh>
#include <Solver.cuh>

namespace clip
{

    NSAllen::NSAllen(InputData idata)
        : m_idata(idata), Solver(idata)
    {

        flagGenLauncher2();

        // initialization();
    };

    void NSAllen::initialization()
    {

        this->allocateOnDevice(dev_f, "dev_f"); // hydrodynamics
        this->allocateOnDevice(dev_g, "dev_g"); // nterface
        this->allocateOnDevice(dev_f_post, "dev_f_post");
        this->allocateOnDevice(dev_g_post, "dev_g_post");

        this->allocateOnDevice(dev_rho, "dev_rho", true);
        this->allocateOnDevice(dev_c, "dev_c", true);
    }

    NSAllen::~NSAllen()
    {

        SAFE_CUDA_FREE(dev_f);
        SAFE_CUDA_FREE(dev_f_post);
        SAFE_CUDA_FREE(dev_g);
        SAFE_CUDA_FREE(dev_g_post);

        SAFE_CUDA_FREE(dev_rho);
        SAFE_CUDA_FREE(dev_rhol);
        SAFE_CUDA_FREE(dev_rhoh);
        SAFE_CUDA_FREE(dev_mul);
        SAFE_CUDA_FREE(dev_muh);
        SAFE_CUDA_FREE(dev_taul);
        SAFE_CUDA_FREE(dev_tauh);
        SAFE_CUDA_FREE(dev_dcdx);
        SAFE_CUDA_FREE(dev_dcdy);
        SAFE_CUDA_FREE(dev_c);
        SAFE_CUDA_FREE(dev_drho3);
        SAFE_CUDA_FREE(dev_c_t);
        SAFE_CUDA_FREE(dev_p);
        SAFE_CUDA_FREE(dev_mu);
        SAFE_CUDA_FREE(dev_ni);
        SAFE_CUDA_FREE(dev_nj);
        SAFE_CUDA_FREE(dev_rhosum);
        SAFE_CUDA_FREE(dev_sigma);
        SAFE_CUDA_FREE(dev_w);
        SAFE_CUDA_FREE(dev_wc);
        SAFE_CUDA_FREE(dev_beta);
        SAFE_CUDA_FREE(dev_kk);
        SAFE_CUDA_FREE(dev_mob);
        SAFE_CUDA_FREE(dev_r0);
        SAFE_CUDA_FREE(dev_gy);
        SAFE_CUDA_FREE(dev_x0);
        SAFE_CUDA_FREE(dev_y0);
        SAFE_CUDA_FREE(dev_u0);
    }

    __global__ void flagGen2()
    {
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

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

        // printf("index:  inside allen ex = %d\n", ex[3]);
    }

    void NSAllen::flagGenLauncher2()
    {

        flagGen2<<<dimGrid, dimBlock>>>();
        cudaDeviceSynchronize();
    }

    __device__ __forceinline__ CLIP_REAL NSAllen::Equilibrium_new(const WMRT::velSet velSet, int q, CLIP_REAL Ux, CLIP_REAL Uy, CLIP_REAL Uz = 0)
    {
        // using namespace nsAllen;
        using namespace WMRT;
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
    __device__ __forceinline__ void NSAllen::calculateVF(const WMRT::velSet velSet, const InputData::SimParams params, CLIP_REAL gneq[q], CLIP_REAL fv[dim], CLIP_REAL tau, CLIP_REAL dcdx, CLIP_REAL dcdy, CLIP_REAL dcdz)
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

    __global__ void KernelInitializeDistributions(const WMRT::velSet velSet, const InputData::SimParams params, const Domain::DomainInfo domain, CLIP_REAL *dev_f, CLIP_REAL *dev_g, CLIP_REAL *dev_f_post, CLIP_REAL *dev_g_post,
                                                  CLIP_REAL *dev_c, CLIP_REAL *dev_rho, CLIP_REAL *dev_p, CLIP_REAL *dev_vel, CLIP_REAL *dev_normal)
    {
        constexpr CLIP_UINT Q = velSet.Q;
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = Domain::getIndex(domain, i, j, k);
        const CLIP_UINT idx_X = Domain::getIndex<DIM>(domain, i, j, k, IDX_X);
        const CLIP_UINT idx_Y = Domain::getIndex<DIM>(domain, i, j, k, IDX_Y);

#ifdef ENABLE_3D
        const CLIP_UINT idx_Z = Domain::getIndex<DIM>(domain, i, j, k, IDX_Z);
#endif

        if (Domain::isInside<DIM>(domain, i, j, k))
        {
            dev_rho[idx_SCALAR] = (s_rhoL) + dev_c[idx_SCALAR] * ((s_rhoH) - (s_rhoL));

            if (s_caseType == CaseType::Bubble)
            {
                dev_p[idx_SCALAR] = dev_p[idx_SCALAR] - dev_c[idx_SCALAR] * s_sigma / s_radius / (dev_rho[idx_SCALAR] / 3.0);
            }
            else if (s_caseType == CaseType::Drop)
            {
                dev_p[idx_SCALAR] = dev_p[idx_SCALAR] + dev_c[idx_SCALAR] * s_sigma / s_radius / (dev_rho[idx_SCALAR] / 3.0);
            }
            else
            {
                dev_p[idx_SCALAR] = 0;
            }
            // dev_p[index] = 0;
            dev_vel[idx_X] = 0;
            dev_vel[idx_Y] = 0;

#ifdef ENABLE_3D
            dev_vel[idx_Z] = 0;
#endif

#pragma unroll
            for (CLIP_UINT q = 0; q < Q; q++)
            {

#ifdef ENABLE_2D
                const CLIP_REAL ga_wa = NSAllen::Equilibrium_new(q, dev_vel[idx_X], dev_vel[idx_Y]);
                const CLIP_REAL hlp = wa[q] * ((1.0 - 4.0 * ((dev_c[idx_SCALAR] - 0.50) * (dev_c[idx_SCALAR] - 0.50))) /
                                               s_interfaceWidth * (ex[q] * dev_normal[idx_X] + ey[q] * dev_normal[idx_Y]));

#elif defined(ENABLE_3D)
                const CLIP_REAL ga_wa = NSAllen::Equilibrium_new(q, dev_vel[idx_X], dev_vel[idx_Y], dev_vel[idx_Z]);
                /// TO DO adding eq for 3D
#endif

                const CLIP_REAL Gamma = ga_wa + wa[q];

                //*******************geq
                dev_g_post[NSAllen::getIndex<Q>(i, j, k, q)] = dev_c[idx_SCALAR] * Gamma - 0.50 * hlp;
                dev_g[NSAllen::getIndex<Q>(i, j, k, q)] = dev_c[idx_SCALAR] * Gamma - 0.50 * hlp;
                //*******************geq
                dev_f_post[NSAllen::getIndex<NSAllen::Q>(i, j, k, q)] = dev_p[idx_SCALAR] * wa[q] + ga_wa;
                dev_f[NSAllen::getIndex<Q>(i, j, k, q)] = dev_p[idx_SCALAR] * wa[q] + ga_wa;
            }
        }
    }

    //     __global__ void Chemical_Potential(double *dev_c, double *dev_mu)
    //     {
    //         using namespace nsAllen;
    //         using namespace WMRT;
    //         const CLIP_UINT i = THREAD_IDX_X;
    //         const CLIP_UINT j = THREAD_IDX_Y;
    //         const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

    //         const CLIP_UINT idx_SCALAR = NSAllen::getIndex(i, j, k);

    //         if (NSAllen::isInside<DIM, true>(i, j, k))
    //         {

    // #ifdef ENABLE_2D
    //             const CLIP_REAL D2C = (dev_c[NSAllen::getIndex(i - 1, j - 1, 0)] + dev_c[NSAllen::getIndex(i + 1, j - 1, 0)] + dev_c[NSAllen::getIndex(i - 1, j + 1, 0)] +
    //                                    dev_c[NSAllen::getIndex(i + 1, j + 1, 0)] + 4.0 * (dev_c[NSAllen::getIndex(i, j - 1, 0)] + dev_c[NSAllen::getIndex(i - 1, j, 0)] + dev_c[NSAllen::getIndex(i + 1, j, 0)] + dev_c[NSAllen::getIndex(i, j + 1, 0)]) - 20 * dev_c[NSAllen::getIndex(i, j, 0)]) /
    //                                   6.0;
    // #elif defined(ENABLE_3D)

    //             const CLIP_REAL D2C = (20.0 * (dev_c[NSAllen::getIndex(i + 1, j, k)] + dev_c[NSAllen::getIndex(i - 1, j, k)] + dev_c[NSAllen::getIndex(i, j + 1, k)] + dev_c[NSAllen::getIndex(i, j - 1, k)] + dev_c[NSAllen::getIndex(i, j, k + 1)] + dev_c[NSAllen::getIndex(i, j, k - 1)]) +
    //                                    6.0 * (dev_c[NSAllen::getIndex(i + 1, j + 1, k)] + dev_c[NSAllen::getIndex(i, j + 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k)] + dev_c[NSAllen::getIndex(i, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j, k - 1)] +
    //                                           dev_c[NSAllen::getIndex(i - 1, j, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j, k + 1)] + dev_c[NSAllen::getIndex(i, j - 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k)] + dev_c[NSAllen::getIndex(i, j - 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k)]) +
    //                                    (dev_c[NSAllen::getIndex(i + 1, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j + 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k - 1)] +
    //                                     dev_c[NSAllen::getIndex(i - 1, j - 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k + 1)]) -
    //                                    200.0 * dev_c[NSAllen::getIndex(i, j, k)]) /
    //                                   48.0;
    // #endif

    //             dev_mu[idx_SCALAR] = 4.0 * s_betaConstant * dev_c[idx_SCALAR] * (dev_c[idx_SCALAR] - 1.0) * (dev_c[idx_SCALAR] - 0.50) - s_kConstant * D2C;
    //         }
    //     }

    //     __global__ void normal_FD(double *dev_dc, double *dev_normal)
    //     {
    //         using namespace nsAllen;
    //         using namespace WMRT;
    //         const CLIP_UINT i = THREAD_IDX_X;
    //         const CLIP_UINT j = THREAD_IDX_Y;
    //         const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

    //         const CLIP_UINT idx_SCALAR = NSAllen::getIndex(i, j, k);
    //         const CLIP_UINT idx_X = NSAllen::getIndex<DIM>(i, j, k, IDX_X);
    //         const CLIP_UINT idx_Y = NSAllen::getIndex<DIM>(i, j, k, IDX_Y);

    // #ifdef ENABLE_3D
    //         const CLIP_UINT idx_Z = NSAllen::getIndex<DIM>(i, j, k, IDX_Z);
    // #endif

    //         if (NSAllen::isInside<DIM, true>(i, j, k))
    //         {

    // #ifdef ENABLE_2D

    //             const CLIP_REAL tmp = sqrt((dev_dc[idx_X] * dev_dc[idx_X]) + (dev_dc[idx_Y] * dev_dc[idx_Y])) + 1e-32;

    //             dev_normal[idx_X] = dev_dc[idx_X] / tmp;
    //             dev_normal[idx_Y] = dev_dc[idx_Y] / tmp;

    // #elif defined(ENABLE_3D)

    //             const CLIP_REAL tmp = sqrt((dev_dc[idx_X] * dev_dc[idx_X]) + (dev_dc[idx_Y] * dev_dc[idx_Y]) + (dev_dc[idx_Z] * dev_dc[idx_Z])) + 1e-32;

    //             dev_normal[idx_X] = dev_dc[idx_X] / tmp;
    //             dev_normal[idx_Y] = dev_dc[idx_Y] / tmp;
    //             dev_normal[idx_Z] = dev_dc[idx_Z] / tmp;
    // #endif
    //         }
    //     }

    //     __global__ void Isotropic_Gradient(double *dev_c, double *dev_dc)
    //     {
    //         using namespace nsAllen;
    //         const CLIP_UINT i = THREAD_IDX_X;
    //         const CLIP_UINT j = THREAD_IDX_Y;
    //         const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

    //         const CLIP_UINT idx_SCALAR = NSAllen::getIndex(i, j, k);
    //         const CLIP_UINT idx_X = NSAllen::getIndex<DIM>(i, j, k, IDX_X);
    //         const CLIP_UINT idx_Y = NSAllen::getIndex<DIM>(i, j, k, IDX_Y);

    // #ifdef ENABLE_3D
    //         const CLIP_UINT idx_Z = NSAllen::getIndex<DIM>(i, j, k, IDX_Z);
    // #endif

    //         if (NSAllen::isInside<DIM, true>(i, j, k))
    //         {

    // #ifdef ENABLE_2D

    //             dev_dc[idx_X] = (dev_c[NSAllen::getIndex(i + 1, j, k)] - dev_c[NSAllen::getIndex(i - 1, j, k)]) / 3.0 +
    //                             (dev_c[NSAllen::getIndex(i + 1, j - 1, k)] + dev_c[NSAllen::getIndex(i + 1, j + 1, k)] - dev_c[NSAllen::getIndex(i - 1, j - 1, k)] - dev_c[NSAllen::getIndex(i - 1, j + 1, k)]) / 12.0;

    //             dev_dc[idx_Y] = (dev_c[NSAllen::getIndex(i, j + 1, k)] - dev_c[NSAllen::getIndex(i, j - 1, k)]) / 3.0 +
    //                             (dev_c[NSAllen::getIndex(i - 1, j + 1, k)] + dev_c[NSAllen::getIndex(i + 1, j + 1, k)] - dev_c[NSAllen::getIndex(i - 1, j - 1, k)] - dev_c[NSAllen::getIndex(i + 1, j - 1, k)]) / 12.0;

    // #elif defined(ENABLE_3D)

    //             dev_dc[idx_X] = (0.50) * ((4.0 / 9.0) * (dev_c[NSAllen::getIndex(i + 1, j, k)] - dev_c[NSAllen::getIndex(i - 1, j, k)]) + (1.0 / 9.0) * ((dev_c[NSAllen::getIndex(i + 1, j, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j, k - 1)] + dev_c[NSAllen::getIndex(i + 1, j + 1, k)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k)]) - (dev_c[NSAllen::getIndex(i - 1, j, k + 1)] + dev_c[NSAllen::getIndex(i - 1, j, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k)])) +
    //                                       (1.0 / 36.0) * ((dev_c[NSAllen::getIndex(i + 1, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k + 1)] + dev_c[getIndex(i + 1, j + 1, k - 1)] + dev_c[getIndex(i + 1, j - 1, k - 1)]) -
    //                                                       (dev_c[NSAllen::getIndex(i - 1, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k + 1)] + dev_c[getIndex(i - 1, j + 1, k - 1)] + dev_c[getIndex(i - 1, j - 1, k - 1)])));

    //             dev_dc[idx_Y] = (0.50) * ((4.0 / 9.0) * (dev_c[NSAllen::getIndex(i, j + 1, k)] - dev_c[NSAllen::getIndex(i, j - 1, k)]) + (1.0 / 9.0) * ((dev_c[NSAllen::getIndex(i + 1, j + 1, k)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k)] + dev_c[NSAllen::getIndex(i, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i, j + 1, k - 1)]) - (dev_c[NSAllen::getIndex(i, j - 1, k + 1)] + dev_c[NSAllen::getIndex(i, j - 1, k - 1)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k)])) +
    //                                       (1.0 / 36.0) * ((dev_c[NSAllen::getIndex(i + 1, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j + 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k - 1)]) -
    //                                                       (dev_c[NSAllen::getIndex(i + 1, j - 1, k + 1)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k - 1)])));

    //             dev_dc[idx_Z] = (0.50) * ((4.0 / 9.0) * (dev_c[NSAllen::getIndex(i, j, k + 1)] - dev_c[NSAllen::getIndex(i, j, k - 1)]) + (1.0 / 9.0) * ((dev_c[NSAllen::getIndex(i + 1, j, k + 1)] + dev_c[NSAllen::getIndex(i - 1, j, k + 1)] + dev_c[NSAllen::getIndex(i, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i, j - 1, k + 1)]) - (dev_c[NSAllen::getIndex(i + 1, j, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j, k - 1)] + dev_c[NSAllen::getIndex(i, j + 1, k - 1)] + dev_c[NSAllen::getIndex(i, j - 1, k - 1)])) +
    //                                       (1.0 / 36.0) * ((dev_c[NSAllen::getIndex(i + 1, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k + 1)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k + 1)]) -
    //                                                       (dev_c[NSAllen::getIndex(i + 1, j + 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k - 1)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k - 1)])));

    // #endif
    //         }
    //     }

    //     __global__ void kernelStreaming(double *f, double *f_post)
    //     {

    //         using namespace nsAllen;
    //         using namespace WMRT;
    //         constexpr CLIP_UINT Q = WMRT::Q;
    //         const CLIP_UINT i = THREAD_IDX_X;
    //         const CLIP_UINT j = THREAD_IDX_Y;
    //         const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

    //         const CLIP_UINT idx_SCALAR = NSAllen::getIndex(i, j, k);
    //         const CLIP_UINT idx_X = NSAllen::getIndex<DIM>(i, j, k, IDX_X);
    //         const CLIP_UINT idx_Y = NSAllen::getIndex<DIM>(i, j, k, IDX_Y);

    // #ifdef ENABLE_3D
    //         const CLIP_UINT idx_Z = NSAllen::getIndex<DIM>(i, j, k, IDX_Z);
    // #endif

    //         if (NSAllen::isInside<DIM, true>(i, j, k))
    //         {
    // #pragma unroll
    //             for (int q = 0; q < Q; q++)
    //             {

    // #ifdef ENABLE_3D

    // #endif

    // #ifdef ENABLE_2D
    //                 const CLIP_UINT id = i - ex[q];
    //                 const CLIP_UINT jd = j - ey[q];
    //                 const CLIP_UINT kd = 0;
    // #elif defined(ENABLE_3D)
    //                 const CLIP_UINT id = i - ex[q];
    //                 const CLIP_UINT jd = j - ey[q];
    //                 const CLIP_UINT kd = k - ez[q];
    // #endif

    //                 // if (id >= 0 && jd >= 0 && kd >= 0 && id < N[0] && jd < N[1] && kd < N[2])
    //                 {
    //                     f_post[NSAllen::getIndex<Q>(i, j, k, q)] = f[NSAllen::getIndex<Q>(id, jd, kd, q)];
    //                 }
    //             }
    //         }
    //     }

    //     __global__ void kernelMacroscopicg(double *dev_p, double *dev_rho, double *dev_c, double *dev_f_post, double *dev_dc, double *dev_vel, double *dev_mu)
    //     {

    //         using namespace nsAllen;
    //         using namespace WMRT;
    //         constexpr CLIP_UINT Q = WMRT::Q;

    //         CLIP_REAL gneq[Q], tmp[Q], fv[DIM], tau;

    //         const CLIP_UINT i = THREAD_IDX_X;
    //         const CLIP_UINT j = THREAD_IDX_Y;
    //         const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

    //         const CLIP_UINT idx_SCALAR = NSAllen::getIndex(i, j, k);
    //         const CLIP_UINT idx_X = NSAllen::getIndex<DIM>(i, j, k, IDX_X);
    //         const CLIP_UINT idx_Y = NSAllen::getIndex<DIM>(i, j, k, IDX_Y);

    // #ifdef ENABLE_3D
    //         const CLIP_UINT idx_Z = NSAllen::getIndex<DIM>(i, j, k, IDX_Z);
    // #endif

    //         if (NSAllen::isInside<DIM, true>(i, j, k))
    //         {

    //             dev_p[idx_SCALAR] = 0;
    // #pragma unroll
    //             for (int q = 0; q < Q; q++)
    //             {
    //                 dev_p[idx_SCALAR] += dev_f_post[NSAllen::getIndex<Q>(i, j, k, q)];
    //             }

    // #pragma unroll

    //             for (int q = 0; q < Q; q++)
    //             {
    // #ifdef ENABLE_2D
    //                 const CLIP_REAL ga_wa = NSAllen::Equilibrium_new(q, dev_vel[idx_X], dev_vel[idx_Y]);

    // #elif defined(ENABLE_3D)
    //                 const CLIP_REAL ga_wa = NSAllen::Equilibrium_new(q, dev_vel[idx_X], dev_vel[idx_Y], dev_vel[idx_Z]);
    // #endif

    //                 const CLIP_REAL geq = dev_p[idx_SCALAR] * wa[q] + ga_wa;
    //                 gneq[q] = dev_f_post[NSAllen::getIndex<Q>(i, j, k, q)] - geq;
    //             }

    // #ifdef ENABLE_2D

    //             Equation::convertD2Q9Weighted(gneq, tmp);

    //             if (dev_c[idx_SCALAR] < 0.0)
    //                 tau = s_tauL;
    //             else if (dev_c[idx_SCALAR] > 1.0)
    //                 tau = s_tauH;
    //             else
    //             {
    //                 tau = s_tauL + dev_c[idx_SCALAR] * (s_tauH - s_tauL);
    //             }
    //             const CLIP_REAL s9 = 1.0 / (tau + 0.50);

    //             tmp[7] = tmp[7] * s9;
    //             tmp[8] = tmp[8] * s9;

    //             Equation::reconvertD2Q9Weighted(tmp, gneq);

    //             NSAllen::calculateVF<Q, DIM>(gneq, fv, tau, dev_dc[idx_X], dev_dc[idx_Y]);
    // #elif defined(ENABLE_3D)
    //             Equation::convertD3Q19Weighted(gneq, tmp);

    //             if (dev_c[idx_SCALAR] < 0.0)
    //                 tau = s_tauL;
    //             else if (dev_c[idx_SCALAR] > 1.0)
    //                 tau = s_tauH;
    //             else
    //             {
    //                 tau = s_tauL + dev_c[idx_SCALAR] * (s_tauH - s_tauL);
    //             }

    //             const CLIP_REAL s9 = 1.0 / (tau + 0.50);

    //             tmp[4] = tmp[4] * s9;
    //             tmp[5] = tmp[5] * s9;
    //             tmp[6] = tmp[6] * s9;
    //             tmp[7] = tmp[7] * s9;
    //             tmp[8] = tmp[8] * s9;

    //             Equation::reconvertD3Q19Weighted(tmp, gneq);

    //             NSAllen::calculateVF<Q, DIM>(gneq, fv, tau, dev_dc[idx_X], dev_dc[idx_Y], dev_dc[idx_Z]);

    // #endif

    //             CLIP_REAL Fgy = 0;
    //             if (s_caseType == CaseType::Bubble)
    //             {
    //                 Fgy = -(dev_rho[idx_SCALAR] - s_rhoH) * s_gravity;
    //             }
    //             else if (s_caseType == CaseType::Drop)
    //             {
    //                 Fgy = -(dev_rho[idx_SCALAR] - s_rhoL) * s_gravity;
    //             }

    //             // double Fgy = (*dev_rhol - dev_rho[index]) * *dev_gy;
    //             //  double Fgy = (-dev_rho[index]) * *dev_gy;

    //             const CLIP_REAL Fpx = -dev_p[idx_SCALAR] * s_drho3 * dev_dc[idx_X];
    //             const CLIP_REAL Fpy = -dev_p[idx_SCALAR] * s_drho3 * dev_dc[idx_Y];

    //             const CLIP_REAL Fx = dev_mu[idx_SCALAR] * dev_dc[idx_X] + Fpx + fv[0];
    //             const CLIP_REAL Fy = dev_mu[idx_SCALAR] * dev_dc[idx_Y] + Fpy + Fgy + fv[1];

    // #ifdef ENABLE_3D
    //             const CLIP_REAL Fpz = -dev_p[idx_SCALAR] * s_drho3 * dev_dc[idx_Z];
    //             const CLIP_REAL Fz = dev_mu[idx_SCALAR] * dev_dc[idx_Z] + Fpz + fv[2];
    // #endif

    //             dev_vel[idx_X] = 0;
    //             dev_vel[idx_Y] = 0;
    // #ifdef ENABLE_3D
    //             dev_vel[idx_Z] = 0;
    // #endif

    // #pragma unroll
    //             for (int q = 0; q < Q; q++)
    //             {
    //                 dev_vel[idx_X] += ex[q] * dev_f_post[NSAllen::getIndex<Q>(i, j, k, q)];
    //                 dev_vel[idx_Y] += ey[q] * dev_f_post[NSAllen::getIndex<Q>(i, j, k, q)];
    // #ifdef ENABLE_3D
    //                 dev_vel[idx_Z] += ez[q] * dev_f_post[NSAllen::getIndex<Q>(i, j, k, q)];
    // #endif
    //             }

    //             dev_vel[idx_X] = dev_vel[idx_X] + 0.50 * Fx / dev_rho[idx_SCALAR];
    //             dev_vel[idx_Y] = dev_vel[idx_Y] + 0.50 * Fy / dev_rho[idx_SCALAR];
    // #ifdef ENABLE_3D
    //             dev_vel[idx_Z] = dev_vel[idx_Z] + 0.50 * Fz / dev_rho[idx_SCALAR];
    // #endif
    //         }
    //     }

    //     __global__ void kernelMacroscopich(double *dev_p, double *dev_g_post, double *dev_rho, double *dev_c, double *dev_rhoh, double *dev_rhol, double *dev_r0, double *dev_x0, double *dev_y0)
    //     {
    //         using namespace nsAllen;
    //         using namespace WMRT;
    //         constexpr CLIP_UINT Q = WMRT::Q;

    //         const CLIP_UINT i = THREAD_IDX_X;
    //         const CLIP_UINT j = THREAD_IDX_Y;
    //         const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

    //         const CLIP_UINT idx_SCALAR = NSAllen::getIndex(i, j, k);

    //         if (NSAllen::isInside<DIM, true>(i, j, k))
    //         {

    //             dev_c[idx_SCALAR] = 0;
    // #pragma unroll
    //             for (CLIP_UINT q = 0; q < Q; q++)
    //             {
    //                 dev_c[idx_SCALAR] += dev_g_post[NSAllen::getIndex<Q>(i, j, k, q)];
    //             }

    //             dev_rho[idx_SCALAR] = (s_rhoL + (dev_c[idx_SCALAR] * (s_rhoH - s_rhoL)));
    //         }
    //     }

    //     __global__ void kernelCollisionMRTh(CLIP_REAL *dev_g, CLIP_REAL *dev_g_post, CLIP_REAL *dev_c, CLIP_REAL *dev_rho, CLIP_REAL *dev_vel, CLIP_REAL *dev_normal)
    //     {

    //         using namespace nsAllen;
    //         using namespace WMRT;
    //         constexpr CLIP_UINT Q = WMRT::Q;

    //         const CLIP_REAL wc = 1.0 / (0.50 + 3.0 * s_interfaceWidth);

    //         const CLIP_UINT i = THREAD_IDX_X;
    //         const CLIP_UINT j = THREAD_IDX_Y;
    //         const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

    //         const CLIP_UINT idx_SCALAR = NSAllen::getIndex(i, j, k);
    //         const CLIP_UINT idx_X = NSAllen::getIndex<DIM>(i, j, k, IDX_X);
    //         const CLIP_UINT idx_Y = NSAllen::getIndex<DIM>(i, j, k, IDX_Y);

    // #ifdef ENABLE_3D
    //         const CLIP_UINT idx_Z = NSAllen::getIndex<DIM>(i, j, k, IDX_Z);
    // #endif

    //         if (NSAllen::isInside<DIM, true>(i, j, k))
    //         {

    // #pragma unroll
    //             for (CLIP_UINT q = 0; q < Q; q++)
    //             {

    // #ifdef ENABLE_2D
    //                 const CLIP_REAL ga_wa = NSAllen::Equilibrium_new(q, dev_vel[idx_X], dev_vel[idx_Y]);
    //                 const CLIP_REAL e_normal = (ex[q] * dev_normal[idx_X] + ey[q] * dev_normal[idx_Y]);
    // #elif defined(ENABLE_3D)
    //                 const CLIP_REAL ga_wa = NSAllen::Equilibrium_new(q, dev_vel[idx_X], dev_vel[idx_Y], dev_vel[idx_Z]);
    //                 const CLIP_REAL e_normal = (ex[q] * dev_normal[idx_X] + ey[q] * dev_normal[idx_Y] + ez[q] * dev_normal[idx_Z]);
    // #endif

    //                 const CLIP_REAL eF = ((3.0 * (s_mobility) * (1.0 - 4.0 * ((dev_c[idx_SCALAR] - 0.50) * (dev_c[idx_SCALAR] - 0.50)))) / (s_interfaceWidth)) * e_normal;
    //                 const CLIP_REAL hlp = wa[q] * eF;
    //                 const CLIP_REAL heq = dev_c[idx_SCALAR] * (ga_wa + wa[q]) + hlp;
    //                 dev_g[NSAllen::getIndex<Q>(i, j, k, q)] = dev_g_post[NSAllen::getIndex<Q>(i, j, k, q)] * (1.0 - (wc)) + heq * (wc);
    //             }
    //         }
    //     }

    //     __global__ void kernelCollisionMRTg(CLIP_REAL *dev_f, CLIP_REAL *dev_f_post, CLIP_REAL *dev_p, CLIP_REAL *dev_c, CLIP_REAL *dev_dc,
    //         CLIP_REAL *dev_mu, CLIP_REAL *dev_rho, CLIP_REAL *dev_vel, CLIP_REAL *dev_normal)
    //     {

    //         using namespace nsAllen;
    //         using namespace WMRT;
    //         constexpr CLIP_UINT Q = WMRT::Q;
    //         CLIP_REAL gneq[Q], tmp[Q], ga_wa[Q], hlp[Q], fv[DIM], tau, s9;

    //         const CLIP_UINT i = THREAD_IDX_X;
    //         const CLIP_UINT j = THREAD_IDX_Y;
    //         const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

    //         const CLIP_UINT idx_SCALAR = NSAllen::getIndex(i, j, k);
    //         const CLIP_UINT idx_X = NSAllen::getIndex<DIM>(i, j, k, IDX_X);
    //         const CLIP_UINT idx_Y = NSAllen::getIndex<DIM>(i, j, k, IDX_Y);

    // #ifdef ENABLE_3D
    //         const CLIP_UINT idx_Z = NSAllen::getIndex<DIM>(i, j, k, IDX_Z);
    // #endif

    //         if (NSAllen::isInside<DIM, true>(i, j, k))
    //         {

    // #pragma unroll

    //             for (CLIP_UINT q = 0; q < Q; q++)
    //             {
    // #ifdef ENABLE_2D
    //                 ga_wa[q] = NSAllen::Equilibrium_new(q, dev_vel[idx_X], dev_vel[idx_Y]);

    // #elif defined(ENABLE_3D)
    //                 ga_wa[q] = NSAllen::Equilibrium_new(q, dev_vel[idx_X], dev_vel[idx_Y], dev_vel[idx_Z]);
    // #endif
    //                 const CLIP_REAL geq = dev_p[idx_SCALAR] * wa[q] + ga_wa[q];
    //                 gneq[q] = dev_f_post[NSAllen::getIndex<Q>(i, j, k, q)] - geq;
    //             }

    // #ifdef ENABLE_2D

    //             Equation::convertD2Q9Weighted(gneq, tmp);

    //             if (dev_c[idx_SCALAR] < 0.0)
    //                 tau = s_tauL;
    //             else if (dev_c[idx_SCALAR] > 1.0)
    //                 tau = s_tauH;
    //             else
    //             {
    //                 tau = s_tauL + dev_c[idx_SCALAR] * (s_tauH - s_tauL);
    //             }
    //             s9 = 1.0 / (tau + 0.50);

    //             tmp[7] = tmp[7] * s9;
    //             tmp[8] = tmp[8] * s9;

    //             Equation::reconvertD2Q9Weighted(tmp, gneq);

    //             NSAllen::calculateVF<Q, DIM>(gneq, fv, tau, dev_dc[idx_X], dev_dc[idx_Y]);
    // #elif defined(ENABLE_3D)
    //             Equation::convertD3Q19Weighted(gneq, tmp);

    //             if (dev_c[idx_SCALAR] < 0.0)
    //                 tau = s_tauL;
    //             else if (dev_c[idx_SCALAR] > 1.0)
    //                 tau = s_tauH;
    //             else
    //             {
    //                 tau = s_tauL + dev_c[idx_SCALAR] * (s_tauH - s_tauL);
    //             }

    //             s9 = 1.0 / (tau + 0.50);

    //             tmp[4] = tmp[4] * s9;
    //             tmp[5] = tmp[5] * s9;
    //             tmp[6] = tmp[6] * s9;
    //             tmp[7] = tmp[7] * s9;
    //             tmp[8] = tmp[8] * s9;

    //             Equation::reconvertD3Q19Weighted(tmp, gneq);

    //             NSAllen::calculateVF<Q, DIM>(gneq, fv, tau, dev_dc[idx_X], dev_dc[idx_Y], dev_dc[idx_Z]);

    // #endif

    // CLIP_REAL Fgy = 0;
    // if (s_caseType == CaseType::Bubble)
    // {
    //     Fgy = -(dev_rho[idx_SCALAR] - s_rhoH) * s_gravity;
    // }
    // else if (s_caseType == CaseType::Drop)
    // {
    //     Fgy = -(dev_rho[idx_SCALAR] - s_rhoL) * s_gravity;
    // }

    // // double Fgy = (*dev_rhol - dev_rho[index]) * *dev_gy;
    // //  double Fgy = (-dev_rho[index]) * *dev_gy;

    // const CLIP_REAL Fpx = -dev_p[idx_SCALAR] * s_drho3 * dev_dc[idx_X];
    // const CLIP_REAL Fpy = -dev_p[idx_SCALAR] * s_drho3 * dev_dc[idx_Y];

    // const CLIP_REAL Fx = dev_mu[idx_SCALAR] * dev_dc[idx_X] + Fpx + fv[0];
    // const CLIP_REAL Fy = dev_mu[idx_SCALAR] * dev_dc[idx_Y] + Fpy + Fgy + fv[1];

    // #ifdef ENABLE_3D
    // const CLIP_REAL Fpz = -dev_p[idx_SCALAR] * s_drho3 * dev_dc[idx_Z];
    // const CLIP_REAL Fz = dev_mu[idx_SCALAR] * dev_dc[idx_Z] + Fpz + fv[2];
    // #endif

    // #pragma unroll
    //             for (CLIP_UINT q = 0; q < Q; q++)
    //             {
    // #ifdef ENABLE_2D
    //                 const CLIP_REAL eF = ex[q] * Fx + ey[q] * Fy;

    // #elif defined(ENABLE_3D)
    //                 const CLIP_REAL eF = ex[q] * Fx + ey[q] * Fy + ez[q] * Fz;
    // #endif
    //                 hlp[q] = 3.0 * wa[q] * eF / dev_rho[idx_SCALAR];
    //                 const CLIP_REAL feq = dev_p[idx_SCALAR] * wa[q] + ga_wa[q] - 0.50 * hlp[q];
    //                 gneq[q] = dev_f_post[NSAllen::getIndex<Q>(i, j, k, q)] - feq;
    //             }

    //             #ifdef ENABLE_2D

    //             Equation::convertD2Q9Weighted(gneq, tmp);

    //             if (dev_c[idx_SCALAR] < 0.0)
    //                 tau = s_tauL;
    //             else if (dev_c[idx_SCALAR] > 1.0)
    //                 tau = s_tauH;
    //             else
    //             {
    //                 tau = s_tauL + dev_c[idx_SCALAR] * (s_tauH - s_tauL);
    //             }
    //             s9 = 1.0 / (tau + 0.50);

    //             tmp[7] = tmp[7] * s9;
    //             tmp[8] = tmp[8] * s9;

    //             Equation::reconvertD2Q9Weighted(tmp, gneq);

    // #elif defined(ENABLE_3D)
    //             Equation::convertD3Q19Weighted(gneq, tmp);

    //             if (dev_c[idx_SCALAR] < 0.0)
    //                 tau = s_tauL;
    //             else if (dev_c[idx_SCALAR] > 1.0)
    //                 tau = s_tauH;
    //             else
    //             {
    //                 tau = s_tauL + dev_c[idx_SCALAR] * (s_tauH - s_tauL);
    //             }

    //             s9 = 1.0 / (tau + 0.50);

    //             tmp[4] = tmp[4] * s9;
    //             tmp[5] = tmp[5] * s9;
    //             tmp[6] = tmp[6] * s9;
    //             tmp[7] = tmp[7] * s9;
    //             tmp[8] = tmp[8] * s9;

    //             Equation::reconvertD3Q19Weighted(tmp, gneq);

    // #endif

    // #pragma unroll
    //             for (int q = 0; q < Q; q++)
    //             {
    //                 dev_f[NSAllen::getIndex<Q>(i, j, k, q)] = dev_f_post[NSAllen::getIndex<Q>(i, j, k, q)] - gneq[q] + hlp[q];
    //             }
    //         }
    //     }

}