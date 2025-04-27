#pragma once
#include "includes.h"
#include "InputData.cuh"
#include "DataArray.cuh"
#include "Domain.cuh"
#include "TimeInfo.cuh"
#include "Boundary.cuh"

namespace clip
{

    class CheckPointer
    {

    public:
        explicit CheckPointer(DataArray &DA, const InputData &idata, const Domain &domain, TimeInfo &ti, const Boundary &boundary);

        virtual ~CheckPointer();
        void save();
        void load();



    private:


        const Domain *m_domain;
        const InputData *m_idata;
        const Boundary *m_boundary;
        TimeInfo *m_ti;
        DataArray *m_DA;
        std::string m_folder;
        std::string m_baseName;




        // template <typename T>
        // void copyToDevice(T*& devPtr, const T* hostPtr, const char* name, CLIP_UINT ndof = SCALAR_FIELD);
        

        template <typename T>
        void saveToFile(T*& hostPtr, const T* devPtr, CLIP_UINT ndof, const std::string& folder, const std::string& name);
        template <typename T>
        void loadFromFile(T*& devPtr, T*& hostPtr, CLIP_UINT ndof, const std::string& folder, const std::string& name);
        void saveTimeInfo(const std::string& folder, const std::string& name);
        void loadTimeInfo(const std::string& folder, const std::string& name);
        void saveDomainSize(const std::string& folder, const std::string& filename);
        void checkDomainSize(const std::string& folder, const std::string& filename);
        void rotateFolders( const std::string& folder);   
        void saveSummaryInfo(const std::string &folder, const std::string &filename);

    };

}
