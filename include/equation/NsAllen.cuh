#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <equation.cuh>

namespace nsAllen
{


    // device variables:
    __constant__ CLIP_REAL s_rhoL;
    __constant__ CLIP_REAL s_rhoH;
    __constant__ CLIP_REAL s_drho3;
    __constant__ CLIP_REAL s_tauL;
    __constant__ CLIP_REAL s_tauH;
    __constant__ CLIP_REAL s_gravity;
    __constant__ CLIP_REAL s_sigma;
    __constant__ CLIP_REAL s_radius;
    __constant__ CLIP_REAL s_interfaceWidth;
    __constant__ CLIP_REAL s_betaConstant;
    __constant__ CLIP_REAL s_kConstant;
    __constant__ CLIP_REAL s_mobility;
    __constant__ CaseType s_caseType;
}

// (double *dev_h, double *dev_g, double *dev_h_post, double *dev_g_post,
//     double *dev_c, double *dev_p, double *dev_ux, double *dev_uy, double *dev_rho, double *dev_ni, double *dev_nj,
//     double *dev_rhol, double *dev_rhoh, double *dev_sigma, double *dev_r0, double *dev_w, double *dev_u0, double *dev_x0,
//     double *dev_y0)

namespace clip
{

    class NSAllen : public Equation
    {

    private:
        void initialization();

    protected:
        // data structures
        InputData m_idata;

        double *host_c, *host_rho, *host_p, *host_vel;

        CLIP_REAL *dev_f, *dev_f_post, *dev_g, *dev_g_post;
        CLIP_REAL *dev_rho, *dev_c, *dev_ux, *dev_uy, *dev_uz;

        double *dev_rhol, *dev_rhoh, *dev_mul, *dev_muh, *dev_taul, *dev_tauh;
        double *dev_dcdx, *dev_dcdy, *dev_drho3, *dev_c_t;
        double *dev_p, *dev_mu, *dev_ni, *dev_nj, *dev_rhosum;
        double *dev_sigma, *dev_w, *dev_wc, *dev_beta, *dev_kk, *dev_mob, *dev_r0, *dev_gy;
        double *dev_x0, *dev_y0, *dev_u0;

        size_t m_nVelocity;


        /// funtions

    public:
        explicit NSAllen(InputData idata);
        ~NSAllen();

        __device__ __forceinline__ static CLIP_REAL Equilibrium_new(int q, CLIP_REAL Ux, CLIP_REAL Uy, CLIP_REAL Uz);

        template <CLIP_UINT q, size_t dim>
        __device__ __forceinline__ static void calculateVF(CLIP_REAL gneq[q], CLIP_REAL fv[dim], CLIP_REAL tau, CLIP_REAL dcdx, CLIP_REAL dcdy, CLIP_REAL dcdz = 0);
        

#ifdef ENABLE_2D
        static constexpr CLIP_UINT Q = 9;
#elif defined(ENABLE_3D)
        static constexpr CLIP_UINT Q = 19;
#endif


    };
}
