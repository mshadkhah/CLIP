#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <Solver.cuh>
#include <DataArray.cuh>


namespace clip
{

    class NSAllen : public Solver
    {

    private:
        void initialization();

    protected:
        // data structures
        InputData m_idata;

        double *host_c, *host_rho, *host_p, *host_vel;

        CLIP_REAL *dev_f, *dev_f_post, *dev_g, *dev_g_post;
        CLIP_REAL *dev_rho, *dev_c, *dev_vel;

        double *dev_rhol, *dev_rhoh, *dev_mul, *dev_muh, *dev_taul, *dev_tauh;
        double *dev_dcdx, *dev_dcdy, *dev_drho3, *dev_c_t;
        double *dev_p, *dev_mu, *dev_ni, *dev_nj, *dev_rhosum;
        double *dev_sigma, *dev_w, *dev_wc, *dev_beta, *dev_kk, *dev_mob, *dev_r0, *dev_gy;
        double *dev_x0, *dev_y0, *dev_u0;

        size_t m_nVelocity;


        /// funtions






    public:
        // explicit NSAllen(InputData idata);


        NSAllen(InputData idata)
        : m_idata(idata), Solver(idata)
    {


        flagGenLauncher2();

        // initialization();
    };




        ~NSAllen();

        __device__ __forceinline__ static CLIP_REAL Equilibrium_new(int q, CLIP_REAL Ux, CLIP_REAL Uy, CLIP_REAL Uz);

        template <CLIP_UINT q, size_t dim>
        __device__ __forceinline__ static void calculateVF(CLIP_REAL gneq[q], CLIP_REAL fv[dim], CLIP_REAL tau, CLIP_REAL dcdx, CLIP_REAL dcdy, CLIP_REAL dcdz = 0);
        



        void flagGenLauncher2 ();

#ifdef ENABLE_2D
        static constexpr CLIP_UINT Q = 9;
#elif defined(ENABLE_3D)
        static constexpr CLIP_UINT Q = 19;
#endif


    };
}
