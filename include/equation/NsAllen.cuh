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
        

    protected:
        // data structures
        InputData m_idata;





        size_t m_nVelocity;


        /// funtions


        Solver::WMRTvelSet m_velset;



    public:
        explicit NSAllen(InputData idata);

        void setVectors();


    //     NSAllen(InputData idata)
    //     : m_idata(idata), Solver(idata)
    // {


    //     flagGenLauncher2();

    //     // initialization();
    // };




        ~NSAllen();

        __device__ __forceinline__ static CLIP_REAL Equilibrium_new(const Solver::WMRTvelSet velSet, int q, CLIP_REAL Ux, CLIP_REAL Uy, CLIP_REAL Uz);

        template <CLIP_UINT q, size_t dim>
        __device__ __forceinline__ static void calculateVF(const Solver::WMRTvelSet velSet, const InputData::SimParams params, CLIP_REAL gneq[q], CLIP_REAL fv[dim], CLIP_REAL tau, CLIP_REAL dcdx, CLIP_REAL dcdy, CLIP_REAL dcdz = 0);
        


        void collision();
        void streaming();
        void macroscopic();
        void solve();
        void initializer();
        void flagGenLauncher2 (const Solver::WMRTvelSet velSet);

// #ifdef ENABLE_2D
//         static constexpr CLIP_UINT Q = 9;
// #elif defined(ENABLE_3D)
//         static constexpr CLIP_UINT Q = 19;
// #endif


    };
}
