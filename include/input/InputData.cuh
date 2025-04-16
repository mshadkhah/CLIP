#pragma once
#include <includes.h>

namespace clip
{

    class InputData
    {
    public:
        explicit InputData(const std::string &filename = "config.txt");
        void read_config();
        std::string getConfig();

        enum CaseType
        {
            Bubble = 0,
            Drop = 1,
            Jet = 2,
            Unknown = 3
        };

        struct SimParams
        {
            // Time and output control

            CLIP_REAL D;
            CLIP_REAL tFinal;
            CLIP_UINT finalStep;
            CLIP_UINT noOutFiles;

            CLIP_UINT N[MAX_DIM];

            CLIP_REAL Bo;
            CLIP_REAL Re;
            CLIP_REAL We;
            CLIP_REAL Pe;
            CLIP_REAL Mo;

            CLIP_REAL rhoRatio;
            CLIP_REAL muRatio;

            CLIP_REAL gravity;
            CLIP_REAL interfaceWidth;
            CLIP_REAL mobility;

            CaseType caseType;
        };

        SimParams params;

    private:
        std::string m_filename;
        template <typename T>
        bool read_value(const std::string &varName, T &var) const;
        template <typename T, std::size_t N>
        bool read_array(const std::string& varName, T (&arr)[N]) const;
        template <typename T>
        bool read_vector(const std::string &varName, std::vector<T> &arr) const;
        static void trim(std::string &s);
        bool read(const std::string &varName, CLIP_REAL &var) const;
        bool read(const std::string &varName, CLIP_UINT &var) const;
        bool read(const std::string &varName, bool &var) const;
        template <typename T, std::size_t N>
        bool read(const std::string &varName, T (&arr)[N]) const;
        bool read(const std::string &varName, CaseType &caseType) const;

        static CaseType caseTypeFromString(const std::string &s);
    };

}
