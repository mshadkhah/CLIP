#include <DataArray.cuh>

namespace clip {

    DataArray::DataArray(InputData idata)
    :m_idata(idata){
        m_nVelocity = m_idata.nVelocity;

    #ifdef ENABLE_2D
        domainDimension = (m_idata.Nx + 2) * (m_idata.Ny + 2);
    #elif defined(ENABLE_3D)
        domainDimension = (m_idata.Nx + 2) * (m_idata.Ny + 2) * (m_idata.Nz + 2);
    #endif
        latticeDimension = domainDimension * m_nVelocity;

    };


    void DataArray::allocateOnDevice(CLIP_REAL* devPtr, const char* name, bool isMacro){
        CLIP_UINT size = 0;
        if (isMacro)
            size = domainDimension;
        else 
            size = latticeDimension;

        cudaMalloc((void **)&devPtr, size * sizeof(CLIP_REAL));
        cudaCheckErrors(("cudaMalloc '" + std::string(name) + "' fail").c_str());
    }



    void DataArray::symbolOnDevice(CLIP_REAL hostVar, CLIP_REAL* devPtr, const char* name, size_t size){

        cudaMemcpyToSymbol(hostVar, &devPtr[0], size * sizeof(CLIP_REAL));
        cudaCheckErrors(("cudaMalloc '" + std::string(name) + "' fail").c_str());
    }


}