#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <equation.cuh>





namespace clip {

    class NSAllen : public Equation{

        private:

        void initialization();
    



        protected:
            InputData m_idata;


            double *host_c, *host_rho, *host_p, *host_ux, *host_uy, *host_uz;
            
            CLIP_REAL *dev_h, *dev_h_post, *dev_g, *dev_g_post;
            double *dev_rho, *dev_rhol, *dev_rhoh, *dev_mul, *dev_muh, *dev_taul, *dev_tauh, *dev_ux, *dev_uy;
            double *dev_dcdx, *dev_dcdy, *dev_c, *dev_drho3, *dev_c_t;
            double *dev_p, *dev_mu, *dev_ni, *dev_nj, *dev_rhosum;
            double *dev_sigma, *dev_w, *dev_wc, *dev_beta, *dev_kk, *dev_mob, *dev_r0, *dev_gy;
            double *dev_x0, *dev_y0, *dev_u0;


            


        public:
            explicit NSAllen(InputData idata);
            ~NSAllen();





            void solve(){

            };



        

        };




        NSAllen::NSAllen(InputData idata)
        :m_idata(idata), Equation(idata){
            initialization();
        };
        


        void NSAllen::initialization(){
            this->allocateOnDevice(dev_h, "dev_h");
        }






        NSAllen::~NSAllen() {
            // Free all device pointers (if non-null)
#define SAFE_CUDA_FREE(ptr) if (ptr) { cudaFree(ptr); ptr = nullptr; }
    
            SAFE_CUDA_FREE(dev_h);
            SAFE_CUDA_FREE(dev_h_post);
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
        }
        


        
    
}



