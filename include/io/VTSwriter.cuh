#pragma once
#include "includes.h"
#include "InputData.cuh"
#include "DataArray.cuh"
#include "Domain.cuh"
#include "TimeInfo.cuh"

namespace clip
{

    class VTSwriter
    {

    public:
        explicit VTSwriter(DataArray &DA, const InputData &idata, const Domain &domain, const TimeInfo &ti, const std::string &folder, const std::string &baseName);

        virtual ~VTSwriter();

        
        void writeToFile();

    private:
        const Domain *m_domain;
        const InputData *m_idata;
        const TimeInfo *m_ti;
        DataArray *m_DA;
        std::string m_folder;
        std::string m_baseName;

        void writeScalar(std::ofstream &file);
        void writeField(std::ofstream &file);
        void writeFieldArray(std::ofstream &file, CLIP_REAL *data, const std::string &name);
        void writeScalarArray(std::ofstream &file, CLIP_REAL *data, const std::string &name);
        void writeVTSBinaryFile();
    };

}
