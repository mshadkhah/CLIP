#pragma once
#include "InputData.cuh"
#include "includes.h"
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
        explicit Boundary(const InputData& idata, const Domain& domain);

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
            Neumann = 4,
            Velocity = 5,
            DoNothing = 6,
            Unknown = 7,
            MAX = 8
        };



        struct Entry
        {
            Objects side = Objects::Unknown;
            Type BCtype = Type::Unknown;
            CLIP_REAL value[MAX_DIM];
            bool ifRefine = false;
        };


        struct BCTypeMap
        {
            Boundary::Type types[static_cast<int>(Boundary::Objects::MAX)];
            CLIP_REAL val[static_cast<int>(Boundary::Objects::MAX)][MAX_DIM];
        };



        __device__ __forceinline__ static bool isMirrorType(Boundary::Type type)
        {
            return (type == Boundary::Type::Wall || 
                    type == Boundary::Type::FreeConvect ||
                    type == Boundary::Type::Neumann ||
                    type == Boundary::Type::SlipWall ||
                    type == Boundary::Type::DoNothing);
        }




        BCTypeMap BCMap;
        bool isPeriodic = false;
        bool isWall = false;
        bool isSlipWall = false;
        bool isFreeConvect = false;
        bool isNeumann = false;
        bool isVelocity = false;



    private:
        const InputData* m_idata;
        const Domain* m_domain;
        dim3 dimBlock, dimGrid;


        std::vector<Entry> boundaries;
        CLIP_UINT boundaryObjects;




        bool readBoundaries(std::vector<Entry>& boundaries);
        void print();
        std::string toString(Type type);
        std::string toString(Objects side);
        Type typeFromString(const std::string& str);
        Objects sideFromString(const std::string& str);
        void trim(std::string &s);
        void updateFlags();
        std::string toLower(const std::string &s);

    };

} // namespace clip
