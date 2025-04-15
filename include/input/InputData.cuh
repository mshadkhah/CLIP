#pragma once
#include <includes.h>


namespace clip {

    class InputData {
        public:
            explicit InputData(const std::string& filename = "config.txt");
            void read_config();
            std::string getConfig();

            CLIP_REAL D;
            CLIP_REAL tFinal;
            CLIP_UINT finalStep;
            CLIP_UINT noOutFiles;
            std::vector<CLIP_UINT> N;
            CLIP_UINT Nx;
            CLIP_UINT Ny;
            CLIP_UINT Nz;
            CLIP_REAL X0;
            CLIP_REAL Y0;
            CLIP_REAL Z0;

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

            CLIP_UINT nVelocity = 9;
        
            

        
        private:
            std::string m_filename;
            template <typename T>
            bool read_value(const std::string& varName, T& var) const;
            template <typename T>
            bool read_array(const std::string& varName, std::vector<T>& arr) const;
            static void trim(std::string& s);
            bool read(const std::string& varName, CLIP_REAL& var) const;
            bool read(const std::string& varName, CLIP_UINT& var) const;
            bool read(const std::string& varName, bool& var) const;
            bool read(const std::string& varName, std::vector<CLIP_UINT>& var) const;
            bool read(const std::string& varName, CaseType& caseType) const;

            static CaseType caseTypeFromString(const std::string& s);
       
        };

    
}



