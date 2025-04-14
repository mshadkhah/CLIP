#include <NsAllen.cuh>

namespace clip
{

    NSAllen::NSAllen(InputData idata)
        : m_idata(idata), Equation(idata)
    {
        initialization();
    };

    void NSAllen::initialization()
    {
        using namespace NSAllen;

        m_nVelocity = m_idata.nVelocity;

#ifdef ENABLE_2D
        m_ex = new CLIP_INT[m_nVelocity]{0, 1, 0, -1, 0, 1, -1, -1, 1};
        m_ey = new CLIP_INT[m_nVelocity]{0, 0, 1, 0, -1, 1, 1, -1, -1};
        m_wa = new CLIP_REAL[m_nVelocity]{4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

        this->symbolOnDevice(ex, m_ex, "ex");
        this->symbolOnDevice(ey, m_ey, "ey");
        this->symbolOnDevice(wa, m_wa, "wa");

#elif defined(ENABLE_3D)
        m_ex = new CLIP_INT[m_nVelocity]{0, 1, 0, -1, 0, 1, -1, -1, 1};
        m_ey = new CLIP_INT[m_nVelocity]{0, 0, 1, 0, -1, 1, 1, -1, -1};
        m_ez = new CLIP_INT[m_nVelocity]{0, 0, 1, 0, -1, 1, 1, -1, -1};
        m_wa = new CLIP_REAL[m_nVelocity]{4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

        this->symbolOnDevice(ex, m_ex, "ex");
        this->symbolOnDevice(ey, m_ey, "ey");
        this->symbolOnDevice(ez, m_ez, "ez");
        this->symbolOnDevice(wa, m_wa, "wa");

#endif

        this->allocateOnDevice(dev_f, "dev_f"); // hydrodynamics
        this->allocateOnDevice(dev_g, "dev_g"); // nterface
        this->allocateOnDevice(dev_f_post, "dev_f_post");
        this->allocateOnDevice(dev_g_post, "dev_g_post");

        this->allocateOnDevice(dev_rho, "dev_rho", true);
        this->allocateOnDevice(dev_c, "dev_c", true);
        this->allocateOnDevice(dev_ux, "dev_ux", true);
        this->allocateOnDevice(dev_uy, "dev_uy", true);

        this->allocateOnDevice(dev_uz, "dev_uz", true);
    }

    NSAllen::~NSAllen()
    {
        // Free all device pointers (if non-null)
#define SAFE_CUDA_FREE(ptr) \
    if (ptr)                \
    {                       \
        cudaFree(ptr);      \
        ptr = nullptr;      \
    }

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
        SAFE_CUDA_FREE(dev_ux);
        SAFE_CUDA_FREE(dev_uy);
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

#undef SAFE_CUDA_FREE

        if (m_ex)
            delete[] m_ex;
        if (m_ey)
            delete[] m_ey;
        if (m_ez)
            delete[] m_ez;
        if (m_wa)
            delete[] m_wa;
    }

    __device__ __forceinline__ CLIP_REAL NSAllen::Equilibrium_new(int q, CLIP_REAL Ux, CLIP_REAL Uy, CLIP_REAL Uz = 0)
    {
        using namespace NSAllen;
        const CLIP_REAL exq = ex[q];
        const CLIP_REAL eyq = ey[q];
        const CLIP_REAL waq = wa[q];

#ifdef ENABLE_2D
        const CLIP_REAL eU = exq * Ux + eyq * Uy;
        const CLIP_REAL U2 = Ux * Ux + Uy * Uy;
#elif defined(ENABLE_3D)
        const CLIP_REAL ezq = ez[q];
        const CLIP_REAL eU = exq * Ux + eyq * Uy + ezq * Uz;
        const CLIP_REAL U2 = Ux * Ux + Uy * Uy + Uz * Uz;
#endif

        return waq * (3.0 * eU + 4.5 * eU * eU - 1.5 * U2);
    }

    __global__ void KernelInitializeDistributions(double *dev_f, double *dev_g, double *dev_f_post, double *dev_g_post,
                                                  double *dev_c, double *dev_rho, double *dev_p, double *dev_vel, double *dev_normal)
    {

        using namespace NSAllen;
        constexpr CLIP_UINT Q = NSAllen::Q;
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = NSAllen::getIndex(i, j, k);
        const CLIP_UINT idx_X = NSAllen::getIndex<DIM>(i, j, k, IDX_X);
        const CLIP_UINT idx_Y = NSAllen::getIndex<DIM>(i, j, k, IDX_Y);

#ifdef ENABLE_3D
        const CLIP_UINT idx_Z = NSAllen::getIndex<DIM>(i, j, k, IDX_Z);
#endif

        if (NSAllen::isInside<DIM>(i, j, k))
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

    __global__ void Chemical_Potential(double *dev_c, double *dev_mu)
    {
        using namespace NSAllen;
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = NSAllen::getIndex(i, j, k);

        if (NSAllen::isInside<DIM, true>(i, j, k))
        {

#ifdef ENABLE_2D
            const CLIP_REAL D2C = (dev_c[NSAllen::getIndex(i - 1, j - 1, 0)] + dev_c[NSAllen::getIndex(i + 1, j - 1, 0)] + dev_c[NSAllen::getIndex(i - 1, j + 1, 0)] +
                                   dev_c[NSAllen::getIndex(i + 1, j + 1, 0)] + 4.0 * (dev_c[NSAllen::getIndex(i, j - 1, 0)] + dev_c[NSAllen::getIndex(i - 1, j, 0)] + dev_c[NSAllen::getIndex(i + 1, j, 0)] + dev_c[NSAllen::getIndex(i, j + 1, 0)]) - 20 * dev_c[NSAllen::getIndex(i, j, 0)]) /
                                  6.0;
#elif defined(ENABLE_3D)

            const CLIP_REAL D2C = (20.0 * (dev_c[NSAllen::getIndex(i + 1, j, k)] + dev_c[NSAllen::getIndex(i - 1, j, k)] + dev_c[NSAllen::getIndex(i, j + 1, k)] + dev_c[NSAllen::getIndex(i, j - 1, k)] + dev_c[NSAllen::getIndex(i, j, k + 1)] + dev_c[NSAllen::getIndex(i, j, k - 1)]) +
                                   6.0 * (dev_c[NSAllen::getIndex(i + 1, j + 1, k)] + dev_c[NSAllen::getIndex(i, j + 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k)] + dev_c[NSAllen::getIndex(i, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j, k - 1)] +
                                          dev_c[NSAllen::getIndex(i - 1, j, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j, k + 1)] + dev_c[NSAllen::getIndex(i, j - 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k)] + dev_c[NSAllen::getIndex(i, j - 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k)]) +
                                   (dev_c[NSAllen::getIndex(i + 1, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j + 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k - 1)] +
                                    dev_c[NSAllen::getIndex(i - 1, j - 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k + 1)]) -
                                   200.0 * dev_c[NSAllen::getIndex(i, j, k)]) /
                                  48.0;
#endif

            dev_mu[idx_SCALAR] = 4.0 * s_betaConstant * dev_c[idx_SCALAR] * (dev_c[idx_SCALAR] - 1.0) * (dev_c[idx_SCALAR] - 0.50) - s_kConstant * D2C;
        }
    }


    __global__ void normal_FD(double *dev_dc, double *dev_normal)
    {
        using namespace NSAllen;
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = NSAllen::getIndex(i, j, k);
        const CLIP_UINT idx_X = NSAllen::getIndex<DIM>(i, j, k, IDX_X);
        const CLIP_UINT idx_Y = NSAllen::getIndex<DIM>(i, j, k, IDX_Y);

#ifdef ENABLE_3D
        const CLIP_UINT idx_Z = NSAllen::getIndex<DIM>(i, j, k, IDX_Z);
#endif

        if (NSAllen::isInside<DIM, true>(i, j, k))
        {

#ifdef ENABLE_2D

            const CLIP_REAL tmp = sqrt((dev_dc[idx_X] * dev_dc[idx_X]) + (dev_dc[idx_Y] * dev_dc[idx_Y])) + 1e-32;

            dev_normal[idx_X] = dev_dc[idx_X] / tmp;
            dev_normal[idx_Y] = dev_dc[idx_Y] / tmp;

#elif defined(ENABLE_3D)

            const CLIP_REAL tmp = sqrt((dev_dc[idx_X] * dev_dc[idx_X]) + (dev_dc[idx_Y] * dev_dc[idx_Y]) + (dev_dc[idx_Z] * dev_dc[idx_Z])) + 1e-32;

            dev_normal[idx_X] = dev_dc[idx_X] / tmp;
            dev_normal[idx_Y] = dev_dc[idx_Y] / tmp;
            dev_normal[idx_Z] = dev_dc[idx_Z] / tmp;
#endif
        }
    }

    __global__ void Isotropic_Gradient(double *dev_c, double *dev_dc)
    {
        using namespace NSAllen;
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

        const CLIP_UINT idx_SCALAR = NSAllen::getIndex(i, j, k);
        const CLIP_UINT idx_X = NSAllen::getIndex<DIM>(i, j, k, IDX_X);
        const CLIP_UINT idx_Y = NSAllen::getIndex<DIM>(i, j, k, IDX_Y);

#ifdef ENABLE_3D
        const CLIP_UINT idx_Z = NSAllen::getIndex<DIM>(i, j, k, IDX_Z);
#endif

        if (NSAllen::isInside<DIM, true>(i, j, k))
        {

#ifdef ENABLE_2D

            dev_dc[idx_X] = (dev_c[NSAllen::getIndex(i + 1, j, k)] - dev_c[NSAllen::getIndex(i - 1, j, k)]) / 3.0 +
                            (dev_c[NSAllen::getIndex(i + 1, j - 1, k)] + dev_c[NSAllen::getIndex(i + 1, j + 1, k)] - dev_c[NSAllen::getIndex(i - 1, j - 1, k)] - dev_c[NSAllen::getIndex(i - 1, j + 1, k)]) / 12.0;

            dev_dc[idx_Y] = (dev_c[NSAllen::getIndex(i, j + 1, k)] - dev_c[NSAllen::getIndex(i, j - 1, k)]) / 3.0 +
                            (dev_c[NSAllen::getIndex(i - 1, j + 1, k)] + dev_c[NSAllen::getIndex(i + 1, j + 1, k)] - dev_c[NSAllen::getIndex(i - 1, j - 1, k)] - dev_c[NSAllen::getIndex(i + 1, j - 1, k)]) / 12.0;

#elif defined(ENABLE_3D)

            dev_dc[idx_X] = (0.50) * ((4.0 / 9.0) * (dev_c[NSAllen::getIndex(i + 1, j, k)] - dev_c[NSAllen::getIndex(i - 1, j, k)]) + (1.0 / 9.0) * ((dev_c[NSAllen::getIndex(i + 1, j, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j, k - 1)] + dev_c[NSAllen::getIndex(i + 1, j + 1, k)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k)]) - (dev_c[NSAllen::getIndex(i - 1, j, k + 1)] + dev_c[NSAllen::getIndex(i - 1, j, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k)])) +
                                        (1.0 / 36.0) * ((dev_c[NSAllen::getIndex(i + 1, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k + 1)] + dev_c[getIndex(i + 1, j + 1, k - 1)] + dev_c[getIndex(i + 1, j - 1, k - 1)]) -
                                                        (dev_c[NSAllen::getIndex(i - 1, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k + 1)] + dev_c[getIndex(i - 1, j + 1, k - 1)] + dev_c[getIndex(i - 1, j - 1, k - 1)])));

            dev_dc[idx_Y] = (0.50) * ((4.0 / 9.0) * (dev_c[NSAllen::getIndex(i, j + 1, k)] - dev_c[NSAllen::getIndex(i, j - 1, k)]) + (1.0 / 9.0) * ((dev_c[NSAllen::getIndex(i + 1, j + 1, k)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k)] + dev_c[NSAllen::getIndex(i, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i, j + 1, k - 1)]) - (dev_c[NSAllen::getIndex(i, j - 1, k + 1)] + dev_c[NSAllen::getIndex(i, j - 1, k - 1)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k)])) +
                                        (1.0 / 36.0) * ((dev_c[NSAllen::getIndex(i + 1, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j + 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k - 1)]) -
                                                        (dev_c[NSAllen::getIndex(i + 1, j - 1, k + 1)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k - 1)])));

            dev_dc[idx_Z] = (0.50) * ((4.0 / 9.0) * (dev_c[NSAllen::getIndex(i, j, k + 1)] - dev_c[NSAllen::getIndex(i, j, k - 1)]) + (1.0 / 9.0) * ((dev_c[NSAllen::getIndex(i + 1, j, k + 1)] + dev_c[NSAllen::getIndex(i - 1, j, k + 1)] + dev_c[NSAllen::getIndex(i, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i, j - 1, k + 1)]) - (dev_c[NSAllen::getIndex(i + 1, j, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j, k - 1)] + dev_c[NSAllen::getIndex(i, j + 1, k - 1)] + dev_c[NSAllen::getIndex(i, j - 1, k - 1)])) +
                                        (1.0 / 36.0) * ((dev_c[NSAllen::getIndex(i + 1, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k + 1)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k + 1)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k + 1)]) -
                                                        (dev_c[NSAllen::getIndex(i + 1, j + 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j + 1, k - 1)] + dev_c[NSAllen::getIndex(i + 1, j - 1, k - 1)] + dev_c[NSAllen::getIndex(i - 1, j - 1, k - 1)])));

#endif
        }
    }



    
__global__ void kernelStreaming(double *f, double *f_post)
{

    using namespace NSAllen;
    constexpr CLIP_UINT Q = NSAllen::Q;
    const CLIP_UINT i = THREAD_IDX_X;
    const CLIP_UINT j = THREAD_IDX_Y;
    const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;

    const CLIP_UINT idx_SCALAR = NSAllen::getIndex(i, j, k);
    const CLIP_UINT idx_X = NSAllen::getIndex<DIM>(i, j, k, IDX_X);
    const CLIP_UINT idx_Y = NSAllen::getIndex<DIM>(i, j, k, IDX_Y);

#ifdef ENABLE_3D
    const CLIP_UINT idx_Z = NSAllen::getIndex<DIM>(i, j, k, IDX_Z);
#endif

    if (NSAllen::isInside<DIM, true>(i, j, k))
    {
#pragma unroll
		for (int q = 0; q < Q; q++)
		{

#ifdef ENABLE_3D

#endif

#ifdef ENABLE_2D
			const CLIP_UINT id = i - ex[q];
			const CLIP_UINT jd = j - ey[q];
            const CLIP_UINT kd = 0;
#elif defined(ENABLE_3D)
            const CLIP_UINT id = i - ex[q];
            const CLIP_UINT jd = j - ey[q];
            const CLIP_UINT kd = k - ez[q];
#endif

			// if (id >= 0 && jd >= 0 && kd >= 0 && id < N[0] && jd < N[1] && kd < N[2])
			{
				f_post[NSAllen::getIndex<Q>(i, j, k, q)] = f[NSAllen::getIndex<Q>(id, jd, kd, q)];
			}
		}
	}
}










}