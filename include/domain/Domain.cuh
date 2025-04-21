#pragma once
#include <includes.h>
#include <InputData.cuh>


namespace clip
{

    class Domain
    {

    public:
        explicit Domain(const InputData& idata);

    
        struct DomainInfo {
            CLIP_UINT extent[MAX_DIM];
            CLIP_UINT domainMinIdx[MAX_DIM];
            CLIP_UINT domainMaxIdx[MAX_DIM];
            CLIP_UINT ghostDomainMinIdx[MAX_DIM];
            CLIP_UINT ghostDomainMaxIdx[MAX_DIM];
        };



        template <CLIP_UINT ndof = 1>
        __host__ __device__ __forceinline__ static CLIP_UINT getIndex(const DomainInfo& domain, CLIP_UINT i, CLIP_UINT j, CLIP_UINT k, CLIP_UINT dof = SCALAR)
        {

            return ((i * domain.extent[IDX_Y] + j) * domain.extent[IDX_Z] + k) * ndof + dof;
        }

        template <CLIP_UINT dim, bool ghosted = false>
        __device__ __forceinline__ static bool isInside(const DomainInfo& domain, CLIP_INT i, CLIP_INT j, CLIP_INT k = 0)
        {
            constexpr CLIP_UINT offset = ghosted ? 1 : 0;

            if constexpr (dim == 2)
            {
                return (i >= offset && i < domain.extent[IDX_X] - offset) &&
                       (j >= offset && j < domain.extent[IDX_Y] - offset);
            }
            else if constexpr (dim == 3)
            {
                return (i >= offset && i < domain.extent[IDX_X] - offset) &&
                       (j >= offset && j < domain.extent[IDX_Y] - offset) &&
                       (k >= offset && k < domain.extent[IDX_Z] - offset);
            }
            else
            {
                return false;
            }
        }


    DomainInfo info;
    CLIP_UINT domainSize;

    private:
        
        CLIP_UINT *m_domainExtent;
        CLIP_UINT *m_domainExtentGhosted;

    protected:
        
        dim3 dimBlock, dimGrid;
        const InputData* m_idata;
    };

}
