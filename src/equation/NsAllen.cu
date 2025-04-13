#include <NsAllen.cuh>

namespace clip
{
    __device__ __forceinline__ CLIP_REAL NSAllen::Equilibrium_new(int q, CLIP_REAL Ux, CLIP_REAL Uy, CLIP_REAL Uz)
    {
        const double exq = ex[q];
        const double eyq = ey[q];
        const double waq = wa[q];

#ifdef ENABLE_2D
        const double eU = exq * Ux + eyq * Uy;
        const double U2 = Ux * Ux + Uy * Uy;
#elif defined(ENABLE_3D)
        const double ezq = ez[q];
        const double eU = exq * Ux + eyq * Uy + ezq * Uz;
        const double U2 = Ux * Ux + Uy * Uy + Uz * Uz;
#endif

        return waq * (3.0 * eU + 4.5 * eU * eU - 1.5 * U2);
    }

    __global__ void KernelInitializeDistributions(double *dev_h, double *dev_g, double *dev_h_post, double *dev_g_post, 
        double *dev_c, double *dev_p, double *dev_ux, double *dev_uy, double *dev_rho, double *dev_ni, double *dev_nj, 
        double *dev_rhol, double *dev_rhoh, double *dev_sigma, double *dev_r0, double *dev_w, double *dev_u0, double *dev_x0, 
        double *dev_y0)
    {
        const int i = THREAD_IDX_X_GHOSTED;
        const int j = THREAD_IDX_Y_GHOSTED;

        int index = getIndex(i, j);

        
        if (i < N[0] && j < N[1])
        {

            dev_rho[index] = (*dev_rhol) + dev_c[index] * ((*dev_rhoh) - (*dev_rhol));

            // dev_p[index] = dev_p[index] + dev_c[index] * *dev_sigma / *dev_r0 / (dev_rho[index] / 3.0);	//in 2D(drop)
            dev_p[index] = dev_p[index] - dev_c[index] * *dev_sigma / *dev_r0 / (dev_rho[index] / 3.0L); // in 2D(bubble)
                                                                                                         // dev_p[index] = 0;
            dev_ux[index] = 0;
            dev_uy[index] = 0;
#pragma unroll
            for (int q = 0; q < 9; q++)
            {
                double ga_wa = Equilibrium_new(dev_ux[index], dev_uy[index], q);
                double Gamma = ga_wa + wa[q];
                double hlp = wa[q] * ((1.0 - 4.0 * ((dev_c[index] - 0.50) * (dev_c[index] - 0.50))) / *dev_w * (ex[q] * dev_ni[index] + ey[q] * dev_nj[index])); // hlp = Wa[i] * eF

                //*******************heq
                dev_h_post[getIndexf(i, j, q)] = dev_c[index] * Gamma - 0.50 * hlp;
                dev_h[getIndexf(i, j, q)] = dev_c[index] * Gamma - 0.50 * hlp;
                //*******************geq
                dev_g_post[getIndexf(i, j, q)] = dev_p[index] * wa[q] + ga_wa;
                dev_g[getIndexf(i, j, q)] = dev_p[index] * wa[q] + ga_wa;
            }
        }
    }

}