#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <Solver.cuh>
#include <DataArray.cuh>
#include "WMRT.cuh"

namespace clip
{

    class NSAllen : public Solver
    {

    public:
        explicit NSAllen(const InputData &idata, const Domain &domain, DataArray &DA, const Boundary &boundary);

        ~NSAllen();

        __device__ __forceinline__ static CLIP_REAL Equilibrium_new(const WMRT::WMRTvelSet velSet, CLIP_UINT q, CLIP_REAL Ux, CLIP_REAL Uy, CLIP_REAL Uz = 0);

        template <CLIP_UINT q, size_t dim>
        __device__ __forceinline__ static void calculateVF(const WMRT::WMRTvelSet velSet, const InputData::SimParams params, CLIP_REAL gneq[q], CLIP_REAL fv[dim], CLIP_REAL tau, CLIP_REAL dcdx, CLIP_REAL dcdy, CLIP_REAL dcdz = 0);
        

        void flagGenLauncher2();
        void solve();
        void initialCondition();
        void deviceInitializer();

    private:
        void initialization();
        
        WMRT::WMRTvelSet m_velset;
        InputData::SimParams m_params;
        Domain::DomainInfo m_info;
        dim3 dimGrid, dimBlock;

        void streaming();
        void collision();
        void macroscopic();
    };
}