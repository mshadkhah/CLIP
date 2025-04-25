#pragma once
#include "InputData.cuh"
#include "includes.h"
#include "DataArray.cuh"
#include "Domain.cuh"


namespace object
{
    constexpr int XMinus = 0;
    constexpr int XPlus = 1;
    constexpr int YMinus = 2;
    constexpr int YPlus = 3;
    constexpr int ZMinus = 4;
    constexpr int ZPlus = 5;
    constexpr int Unknown = 6;
}

namespace clip
{

    class Boundary
    {
    public:
        explicit Boundary(const InputData& idata, const Domain& domain, DataArray& DA);

        ~Boundary();


            enum class Objects {
                XMinus  = 0,
                XPlus   = 1,
                YMinus  = 2,
                YPlus   = 3,
                ZMinus  = 4,
                ZPlus   = 5,
                Unknown = 6,
                MAX     = 7
            };


        enum class Type
        {
            Wall = 0,
            SlipWall = 1,
            FreeConvect = 2,
            Periodic = 3,
            Unknown = 4,
            MAX = 5
        };



        struct Entry
        {
            Objects side = Objects::Unknown;
            Type BCtype = Type::Unknown;
            CLIP_REAL value = 0.0;
            bool ifRefine = false;
        };


        struct BCTypeMap
        {
            Boundary::Type types[static_cast<int>(Boundary::Objects::MAX)];
        };

        BCTypeMap BCMap;


        bool isPeriodic = false;
        bool isWall = false;
        bool isSlipWall = false;

    private:
        const InputData* m_idata;
        const Domain* m_domain;
        DataArray* m_DA;
        dim3 dimBlock, dimGrid;



        void updateFlags();
        std::vector<Entry> boundaries;
        CLIP_UINT boundaryObjects;
        Type* dev_boundaryFlags;




        bool readBoundaries(std::vector<Entry>& boundaries);
        void print();
        std::string toString(Type type);
        std::string toString(Objects side);
        Type typeFromString(const std::string& str);
        Objects sideFromString(const std::string& str);
        void trim(std::string &s);
        void flagGenLauncher(CLIP_UINT* dev_flag, const Domain::DomainInfo& domain);

    };

} // namespace clip
