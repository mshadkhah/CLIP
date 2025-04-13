#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <equation.cuh>

namespace NSAllen
{
    static constexpr CLIP_UINT MAX_Q = 32;
    __constant__ CLIP_REAL wa[MAX_Q];
    __constant__ CLIP_INT ex[MAX_Q];
    __constant__ CLIP_INT ey[MAX_Q];
#ifdef ENABLE_3D
    __constant__ CLIP_UINT ez[MAX_Q];
#endif
}

namespace clip
{

    class NSAllen : public Equation
    {

    private:
        void initialization();

    protected:
        // data structures
        InputData m_idata;

        double *host_c, *host_rho, *host_p, *host_ux, *host_uy, *host_uz;

        CLIP_REAL *dev_f, *dev_f_post, *dev_g, *dev_g_post;
        CLIP_REAL *dev_rho, *dev_c, *dev_ux, *dev_uy, *dev_uz;

        double *dev_rhol, *dev_rhoh, *dev_mul, *dev_muh, *dev_taul, *dev_tauh;
        double *dev_dcdx, *dev_dcdy, *dev_drho3, *dev_c_t;
        double *dev_p, *dev_mu, *dev_ni, *dev_nj, *dev_rhosum;
        double *dev_sigma, *dev_w, *dev_wc, *dev_beta, *dev_kk, *dev_mob, *dev_r0, *dev_gy;
        double *dev_x0, *dev_y0, *dev_u0;

        size_t m_nVelocity;
        CLIP_INT *m_ex;
        CLIP_INT *m_ey;
        CLIP_INT *m_ez;
        CLIP_REAL *m_wa;

        /// funtions
        __device__ __forceinline__ CLIP_REAL Equilibrium_new(int q, CLIP_REAL Ux, CLIP_REAL Uy, CLIP_REAL Uz);

    public:
        explicit NSAllen(InputData idata);

        ~NSAllen();

        void solve() {

        };
    };

    NSAllen::NSAllen(InputData idata)
        : m_idata(idata), Equation(idata)
    {
        initialization();
    };

    void NSAllen::initialization()
    {

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

}
