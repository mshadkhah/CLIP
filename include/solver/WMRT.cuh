#pragma once
#include <includes.h>
#include <InputData.cuh>

namespace clip
{

    class WMRT
    {

    public:
        // explicit WMRT();

        WMRT() {}

        virtual ~WMRT();

        __device__ __forceinline__ static void convertD2Q9Weighted(const CLIP_REAL in[9], CLIP_REAL out[9])
        {
            const CLIP_REAL in0 = in[0], in1 = in[1], in2 = in[2];
            const CLIP_REAL in3 = in[3], in4 = in[4], in5 = in[5];
            const CLIP_REAL in6 = in[6], in7 = in[7], in8 = in[8];

            out[0] = in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8;
            out[1] = -4.0 * in0 - in1 - in2 - in3 - in4 + 2.0 * in5 + 2.0 * in6 + 2.0 * in7 + 2.0 * in8;
            out[2] = 4.0 * in0 - 2.0 * in1 - 2.0 * in2 - 2.0 * in3 - 2.0 * in4 + in5 + in6 + in7 + in8;
            out[3] = in1 - in3 + in5 - in6 - in7 + in8;
            out[4] = -2.0 * in1 + 2.0 * in3 + in5 - in6 - in7 + in8;
            out[5] = in2 - in4 + in5 + in6 - in7 - in8;
            out[6] = -2.0 * in2 + 2.0 * in4 + in5 + in6 - in7 - in8;
            out[7] = in1 - in2 + in3 - in4;
            out[8] = in5 - in6 + in7 - in8;
        }

        __device__ __forceinline__ static void reconvertD2Q9Weighted(const CLIP_REAL in[9], CLIP_REAL out[9])
        {
            const CLIP_REAL in0 = in[0], in1 = in[1], in2 = in[2];
            const CLIP_REAL in3 = in[3], in4 = in[4], in5 = in[5];
            const CLIP_REAL in6 = in[6], in7 = in[7], in8 = in[8];

            out[0] = (4.0 * in0 - 4.0 * in1 + 4.0 * in2) / 36.0;
            out[1] = (4.0 * in0 - in1 - 2.0 * in2 + 6.0 * in3 - 6.0 * in4 + 9.0 * in7) / 36.0;
            out[2] = (4.0 * in0 - in1 - 2.0 * in2 + 6.0 * in5 - 6.0 * in6 - 9.0 * in7) / 36.0;
            out[3] = (4.0 * in0 - in1 - 2.0 * in2 - 6.0 * in3 + 6.0 * in4 + 9.0 * in7) / 36.0;
            out[4] = (4.0 * in0 - in1 - 2.0 * in2 - 6.0 * in5 + 6.0 * in6 - 9.0 * in7) / 36.0;
            out[5] = (4.0 * in0 + 2.0 * in1 + in2 + 6.0 * in3 + 3.0 * in4 + 6.0 * in5 + 3.0 * in6 + 9.0 * in8) / 36.0;
            out[6] = (4.0 * in0 + 2.0 * in1 + in2 - 6.0 * in3 - 3.0 * in4 + 6.0 * in5 + 3.0 * in6 - 9.0 * in8) / 36.0;
            out[7] = (4.0 * in0 + 2.0 * in1 + in2 - 6.0 * in3 - 3.0 * in4 - 6.0 * in5 - 3.0 * in6 + 9.0 * in8) / 36.0;
            out[8] = (4.0 * in0 + 2.0 * in1 + in2 + 6.0 * in3 + 3.0 * in4 - 6.0 * in5 - 3.0 * in6 - 9.0 * in8) / 36.0;
        }

        __device__ __forceinline__ static void convertD3Q19Weighted(const CLIP_REAL in[19], CLIP_REAL out[19])
        {
            const CLIP_REAL in0 = in[0], in1 = in[1], in2 = in[2], in3 = in[3], in4 = in[4],
                            in5 = in[5], in6 = in[6], in7 = in[7], in8 = in[8], in9 = in[9],
                            in10 = in[10], in11 = in[11], in12 = in[12], in13 = in[13], in14 = in[14],
                            in15 = in[15], in16 = in[16], in17 = in[17], in18 = in[18];

            out[0] = in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8 + in9 + in10 + in11 + in12 + in13 + in14 + in15 + in16 + in17 + in18;
            out[1] = in1 - in2 + in7 - in8 + in9 - in10 + in11 - in12 + in13 - in14;
            out[2] = in3 - in4 + in7 - in8 - in9 + in10 + in15 - in16 + in17 - in18;
            out[3] = in5 - in6 + in11 - in12 - in13 + in14 + in15 - in16 - in17 + in18;
            out[4] = in7 + in8 - in9 - in10;
            out[5] = in15 + in16 - in17 - in18;
            out[6] = in11 + in12 - in13 - in14;
            out[7] = 2.0 * in1 + 2.0 * in2 - in3 - in4 - in5 - in6 + in7 + in8 + in9 + in10 + in11 + in12 + in13 + in14 - 2.0 * in15 - 2.0 * in16 - 2.0 * in17 - 2.0 * in18;
            out[8] = in3 + in4 - in5 - in6 + in7 + in8 + in9 + in10 - in11 - in12 - in13 - in14;
            out[9] = -in0 + in7 + in8 + in9 + in10 + in11 + in12 + in13 + in14 + in15 + in16 + in17 + in18;
            out[10] = -2.0 * in1 + 2.0 * in2 + in7 - in8 + in9 - in10 + in11 - in12 + in13 - in14;
            out[11] = -2.0 * in3 + 2.0 * in4 + in7 - in8 - in9 + in10 + in15 - in16 + in17 - in18;
            out[12] = -2.0 * in5 + 2.0 * in6 + in11 - in12 - in13 + in14 + in15 - in16 - in17 + in18;
            out[13] = in7 - in8 + in9 - in10 - in11 + in12 - in13 + in14;
            out[14] = -in7 + in8 + in9 - in10 + in15 - in16 + in17 - in18;
            out[15] = in11 - in12 - in13 + in14 - in15 + in16 + in17 - in18;
            out[16] = in0 - 2.0 * in1 - 2.0 * in2 - 2.0 * in3 - 2.0 * in4 - 2.0 * in5 - 2.0 * in6 + in7 + in8 + in9 + in10 + in11 + in12 + in13 + in14 + in15 + in16 + in17 + in18;
            out[17] = -2.0 * in1 - 2.0 * in2 + in3 + in4 + in5 + in6 + in7 + in8 + in9 + in10 + in11 + in12 + in13 + in14 - 2.0 * in15 - 2.0 * in16 - 2.0 * in17 - 2.0 * in18;
            out[18] = -in3 - in4 + in5 + in6 + in7 + in8 + in9 + in10 - in11 - in12 - in13 - in14;
        }

        __device__ __forceinline__ static void reconvertD3Q19Weighted(const CLIP_REAL in[19], CLIP_REAL out[19])
        {
            const CLIP_REAL in0 = in[0], in1 = in[1], in2 = in[2], in3 = in[3], in4 = in[4];
            const CLIP_REAL in5 = in[5], in6 = in[6], in7 = in[7], in8 = in[8], in9 = in[9];
            const CLIP_REAL in10 = in[10], in11 = in[11], in12 = in[12], in13 = in[13], in14 = in[14];
            const CLIP_REAL in15 = in[15], in16 = in[16], in17 = in[17], in18 = in[18];

            out[0] = (2.0 * in0 - 3.0 * in9 + in16) / 6.0;
            out[1] = (2.0 * in0 + 6.0 * in1 + 3.0 * in7 - 6.0 * in10 - 2.0 * in16 - 3.0 * in17) / 36.0;
            out[2] = (2.0 * in0 - 6.0 * in1 + 3.0 * in7 + 6.0 * in10 - 2.0 * in16 - 3.0 * in17) / 36.0;
            out[3] = (4.0 * in0 + 12.0 * in2 - 3.0 * in7 + 9.0 * in8 - 12.0 * in11 - 4.0 * in16 + 3.0 * in17 - 9.0 * in18) / 72.0;
            out[4] = (4.0 * in0 - 12.0 * in2 - 3.0 * in7 + 9.0 * in8 + 12.0 * in11 - 4.0 * in16 + 3.0 * in17 - 9.0 * in18) / 72.0;
            out[5] = (4.0 * in0 + 12.0 * in3 - 3.0 * in7 - 9.0 * in8 - 12.0 * in12 - 4.0 * in16 + 3.0 * in17 + 9.0 * in18) / 72.0;
            out[6] = (4.0 * in0 - 12.0 * in3 - 3.0 * in7 - 9.0 * in8 + 12.0 * in12 - 4.0 * in16 + 3.0 * in17 + 9.0 * in18) / 72.0;
            out[7] = (4.0 * in0 + 12.0 * in1 + 12.0 * in2 + 36.0 * in4 + 3.0 * in7 + 9.0 * in8 + 6.0 * in9 + 6.0 * in10 + 6.0 * in11 +
                      18.0 * in13 - 18.0 * in14 + 2.0 * in16 + 3.0 * in17 + 9.0 * in18) /
                     144.0;
            out[8] = (4.0 * in0 - 12.0 * in1 - 12.0 * in2 + 36.0 * in4 + 3.0 * in7 + 9.0 * in8 + 6.0 * in9 - 6.0 * in10 - 6.0 * in11 -
                      18.0 * in13 + 18.0 * in14 + 2.0 * in16 + 3.0 * in17 + 9.0 * in18) /
                     144.0;
            out[9] = (4.0 * in0 + 12.0 * in1 - 12.0 * in2 - 36.0 * in4 + 3.0 * in7 + 9.0 * in8 + 6.0 * in9 + 6.0 * in10 - 6.0 * in11 +
                      18.0 * in13 + 18.0 * in14 + 2.0 * in16 + 3.0 * in17 + 9.0 * in18) /
                     144.0;
            out[10] = (4.0 * in0 - 12.0 * in1 + 12.0 * in2 - 36.0 * in4 + 3.0 * in7 + 9.0 * in8 + 6.0 * in9 - 6.0 * in10 + 6.0 * in11 -
                       18.0 * in13 - 18.0 * in14 + 2.0 * in16 + 3.0 * in17 + 9.0 * in18) /
                      144.0;
            out[11] = (4.0 * in0 + 12.0 * in1 + 12.0 * in3 + 36.0 * in6 + 3.0 * in7 - 9.0 * in8 + 6.0 * in9 + 6.0 * in10 + 6.0 * in12 -
                       18.0 * in13 + 18.0 * in15 + 2.0 * in16 + 3.0 * in17 - 9.0 * in18) /
                      144.0;
            out[12] = (4.0 * in0 - 12.0 * in1 - 12.0 * in3 + 36.0 * in6 + 3.0 * in7 - 9.0 * in8 + 6.0 * in9 - 6.0 * in10 - 6.0 * in12 +
                       18.0 * in13 - 18.0 * in15 + 2.0 * in16 + 3.0 * in17 - 9.0 * in18) /
                      144.0;
            out[13] = (4.0 * in0 + 12.0 * in1 - 12.0 * in3 - 36.0 * in6 + 3.0 * in7 - 9.0 * in8 + 6.0 * in9 + 6.0 * in10 - 6.0 * in12 -
                       18.0 * in13 - 18.0 * in15 + 2.0 * in16 + 3.0 * in17 - 9.0 * in18) /
                      144.0;
            out[14] = (4.0 * in0 - 12.0 * in1 + 12.0 * in3 - 36.0 * in6 + 3.0 * in7 - 9.0 * in8 + 6.0 * in9 - 6.0 * in10 + 6.0 * in12 +
                       18.0 * in13 + 18.0 * in15 + 2.0 * in16 + 3.0 * in17 - 9.0 * in18) /
                      144.0;
            out[15] = (4.0 * in0 + 12.0 * in2 + 12.0 * in3 + 36.0 * in5 - 6.0 * in7 + 6.0 * in9 + 6.0 * in11 + 6.0 * in12 +
                       18.0 * in14 - 18.0 * in15 + 2.0 * in16 - 6.0 * in17) /
                      144.0;
            out[16] = (4.0 * in0 - 12.0 * in2 - 12.0 * in3 + 36.0 * in5 - 6.0 * in7 + 6.0 * in9 - 6.0 * in11 - 6.0 * in12 -
                       18.0 * in14 + 18.0 * in15 + 2.0 * in16 - 6.0 * in17) /
                      144.0;
            out[17] = (4.0 * in0 + 12.0 * in2 - 12.0 * in3 - 36.0 * in5 - 6.0 * in7 + 6.0 * in9 + 6.0 * in11 - 6.0 * in12 +
                       18.0 * in14 + 18.0 * in15 + 2.0 * in16 - 6.0 * in17) /
                      144.0;
            out[18] = (4.0 * in0 - 12.0 * in2 + 12.0 * in3 - 36.0 * in5 - 6.0 * in7 + 6.0 * in9 - 6.0 * in11 + 6.0 * in12 -
                       18.0 * in14 - 18.0 * in15 + 2.0 * in16 - 6.0 * in17) /
                      144.0;
        }

        struct WMRTvelSet
        {

#ifdef ENABLE_2D
            static constexpr CLIP_UINT Q = 9;
            const CLIP_INT ex[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
            const CLIP_INT ey[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
            const CLIP_REAL wa[Q] = {4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

#elif defined(ENABLE_3D)
            static constexpr CLIP_UINT Q = 19;
            const CLIP_INT ex[Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
            const CLIP_INT ey[Q] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
            const CLIP_INT ez[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
            const CLIP_REAL wa[Q] = {1.0 / 3.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 36.0,
                                     1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};
#endif
        };

        struct wallBCMap
        {

#ifdef ENABLE_2D
            static constexpr CLIP_UINT Q = 3;
            const CLIP_INT XMinus[Q] = {1, 5, 8};
            const CLIP_INT XPlus[Q] = {3, 7, 6};
            const CLIP_INT YMinus[Q] = {2, 5, 6};
            const CLIP_INT YPlus[Q] = {4, 7, 8};

#elif defined(ENABLE_3D)

            static constexpr CLIP_UINT Q = 5;
            const CLIP_INT XMinus[Q] = {1, 9, 7, 13, 11};
            const CLIP_INT XPlus[Q] = {2, 10, 8, 14, 12};
            const CLIP_INT YMinus[Q] = {3, 10, 7, 17, 15};
            const CLIP_INT YPlus[Q] = {4, 9, 8, 18, 16};
            const CLIP_INT ZMinus[Q] = {5, 14, 11, 18, 15};
            const CLIP_INT ZPlus[Q] = {6, 13, 12, 17, 16};

#endif
        };

        struct slipWallBCMap
        {

#ifdef ENABLE_2D
            static constexpr CLIP_UINT Q = 3;
            const CLIP_INT XMinus[Q] = {1, 5, 8};
            const CLIP_INT XPlus[Q] = {3, 6, 7};
            const CLIP_INT YMinus[Q] = {2, 5, 6};
            const CLIP_INT YPlus[Q] = {4, 8, 7};

#elif defined(ENABLE_3D)

            static constexpr CLIP_UINT Q = 5;
            const CLIP_INT XMinus[Q] = {1, 9, 7, 13, 11};
            const CLIP_INT XPlus[Q] = {2, 8, 10, 12, 14};
            const CLIP_INT YMinus[Q] = {3, 10, 7, 17, 15};
            const CLIP_INT YPlus[Q] = {4, 8, 9, 16, 18};
            const CLIP_INT ZMinus[Q] = {5, 14, 11, 18, 15};
            const CLIP_INT ZPlus[Q] = {6, 12, 13, 16, 17};

#endif
        };


        struct velocityBCMap
        {

#ifdef ENABLE_2D
            static constexpr CLIP_UINT A = 2;
            const CLIP_INT Y[A] = {2, 4};
            const CLIP_INT X[A] = {1, 3};
#elif defined(ENABLE_3D)

            static constexpr CLIP_UINT A = 6;
            const CLIP_INT XZ[A] = {1, 11, 13, 2, 12, 14};
            const CLIP_INT ZX[A] = {5, 11, 14, 6, 12, 13};
            const CLIP_INT YZ[A] = {3, 15, 17, 4, 16, 18};
            const CLIP_INT ZY[A] = {5, 15, 18, 6, 16, 17};
            const CLIP_INT YX[A] = {3, 7, 10, 4, 8, 9};
            const CLIP_INT XY[A] = {1, 7, 9, 2, 8, 10};

#endif
        };






        struct wallCornerBCMap
        {

#ifdef ENABLE_2D
            static constexpr CLIP_UINT Q = 3;
            const CLIP_INT C000[Q] = {5, 1, 2};
            const CLIP_INT C100[Q] = {6, 2, 3};
            const CLIP_INT C110[Q] = {7, 3, 4};
            const CLIP_INT C010[Q] = {8, 4, 1};

#elif defined(ENABLE_3D)

            static constexpr CLIP_UINT Q = 5;
            const CLIP_INT XMinus[Q] = {1, 9, 7, 13, 11};
            const CLIP_INT XPlus[Q] = {2, 8, 10, 12, 14};
            const CLIP_INT YMinus[Q] = {3, 10, 7, 17, 15};
            const CLIP_INT YPlus[Q] = {4, 8, 9, 16, 18};
            const CLIP_INT ZMinus[Q] = {5, 14, 11, 18, 15};
            const CLIP_INT ZPlus[Q] = {6, 12, 13, 16, 17};

#endif
        };




    };

}
