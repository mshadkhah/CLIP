#pragma once
#include <includes.h>
#include <InputData.cuh>
#include <equation.cuh>
#include <DataArray.cuh>







namespace clip {

    class Equation : public DataArray{

        public:
            explicit Equation(InputData idata);
            virtual ~Equation();





        


            __device__ __forceinline__ static void convertD2Q9Weighted(const CLIP_REAL in[9], CLIP_REAL out[9]) {
                const CLIP_REAL in0 = in[0], in1 = in[1], in2 = in[2];
                const CLIP_REAL in3 = in[3], in4 = in[4], in5 = in[5];
                const CLIP_REAL in6 = in[6], in7 = in[7], in8 = in[8];
            
                out[0] = in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8;
            
                out[1] = -4.0 * in0 - in1 - in2 - in3 - in4 + 2.0 * in5 + 2.0 * in6 + 2.0 * in7 + 2.0 * in8;
            
                out[2] =  4.0 * in0 - 2.0 * in1 - 2.0 * in2 - 2.0 * in3 - 2.0 * in4 + in5 + in6 + in7 + in8;
            
                out[3] =  in1 - in3 + in5 - in6 - in7 + in8;
            
                out[4] = -2.0 * in1 + 2.0 * in3 + in5 - in6 - in7 + in8;
            
                out[5] =  in2 - in4 + in5 + in6 - in7 - in8;
            
                out[6] = -2.0 * in2 + 2.0 * in4 + in5 + in6 - in7 - in8;
            
                out[7] =  in1 - in2 + in3 - in4;
            
                out[8] =  in5 - in6 + in7 - in8;
            }
        
        
        
            __device__ __forceinline__ static void reconvertD2Q9Weighted(const CLIP_REAL in[9], CLIP_REAL out[9]) {
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






            __device__ __forceinline__ void convertD3Q19Weighted(const CLIP_REAL in[19], CLIP_REAL out[19]) {
                const CLIP_REAL in0 = in[0],  in1 = in[1],  in2 = in[2],  in3 = in[3],  in4 = in[4],
                             in5 = in[5],  in6 = in[6],  in7 = in[7],  in8 = in[8],  in9 = in[9],
                             in10 = in[10], in11 = in[11], in12 = in[12], in13 = in[13], in14 = in[14],
                             in15 = in[15], in16 = in[16], in17 = in[17], in18 = in[18];
            
                out[0]  = in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8 + in9 + in10 + in11 + in12 + in13 + in14 + in15 + in16 + in17 + in18;
                out[1]  = in1 - in2 + in7 - in8 + in9 - in10 + in11 - in12 + in13 - in14;
                out[2]  = in3 - in4 + in7 - in8 - in9 + in10 + in15 - in16 + in17 - in18;
                out[3]  = in5 - in6 + in11 - in12 - in13 + in14 + in15 - in16 - in17 + in18;
                out[4]  = in7 + in8 - in9 - in10;
                out[5]  = in15 + in16 - in17 - in18;
                out[6]  = in11 + in12 - in13 - in14;
                out[7]  = 2.0 * in1 + 2.0 * in2 - in3 - in4 - in5 - in6 + in7 + in8 + in9 + in10 + in11 + in12 + in13 + in14 - 2.0 * in15 - 2.0 * in16 - 2.0 * in17 - 2.0 * in18;
                out[8]  = in3 + in4 - in5 - in6 + in7 + in8 + in9 + in10 - in11 - in12 - in13 - in14;
                out[9]  = -in0 + in7 + in8 + in9 + in10 + in11 + in12 + in13 + in14 + in15 + in16 + in17 + in18;
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
            



            __device__ __forceinline__ void reconvertD3Q19Weighted(const CLIP_REAL in[19], CLIP_REAL out[19]) {
                const CLIP_REAL in0  = in[0],  in1  = in[1],  in2  = in[2],  in3  = in[3],  in4  = in[4];
                const CLIP_REAL in5  = in[5],  in6  = in[6],  in7  = in[7],  in8  = in[8],  in9  = in[9];
                const CLIP_REAL in10 = in[10], in11 = in[11], in12 = in[12], in13 = in[13], in14 = in[14];
                const CLIP_REAL in15 = in[15], in16 = in[16], in17 = in[17], in18 = in[18];
            
                out[0]  = (2.0L * in0 - 3.0L * in9 + in16) / 6.0L;
                out[1]  = (2.0L * in0 + 6.0L * in1 + 3.0L * in7 - 6.0L * in10 - 2.0L * in16 - 3.0L * in17) / 36.0L;
                out[2]  = (2.0L * in0 - 6.0L * in1 + 3.0L * in7 + 6.0L * in10 - 2.0L * in16 - 3.0L * in17) / 36.0L;
                out[3]  = (4.0L * in0 + 12.0L * in2 - 3.0L * in7 + 9.0L * in8 - 12.0L * in11 - 4.0L * in16 + 3.0L * in17 - 9.0L * in18) / 72.0L;
                out[4]  = (4.0L * in0 - 12.0L * in2 - 3.0L * in7 + 9.0L * in8 + 12.0L * in11 - 4.0L * in16 + 3.0L * in17 - 9.0L * in18) / 72.0L;
                out[5]  = (4.0L * in0 + 12.0L * in3 - 3.0L * in7 - 9.0L * in8 - 12.0L * in12 - 4.0L * in16 + 3.0L * in17 + 9.0L * in18) / 72.0L;
                out[6]  = (4.0L * in0 - 12.0L * in3 - 3.0L * in7 - 9.0L * in8 + 12.0L * in12 - 4.0L * in16 + 3.0L * in17 + 9.0L * in18) / 72.0L;
                out[7]  = (4.0L * in0 + 12.0L * in1 + 12.0L * in2 + 36.0L * in4 + 3.0L * in7 + 9.0L * in8 + 6.0L * in9 + 6.0L * in10 + 6.0L * in11 +
                           18.0L * in13 - 18.0L * in14 + 2.0L * in16 + 3.0L * in17 + 9.0L * in18) / 144.0L;
                out[8]  = (4.0L * in0 - 12.0L * in1 - 12.0L * in2 + 36.0L * in4 + 3.0L * in7 + 9.0L * in8 + 6.0L * in9 - 6.0L * in10 - 6.0L * in11 -
                           18.0L * in13 + 18.0L * in14 + 2.0L * in16 + 3.0L * in17 + 9.0L * in18) / 144.0L;
                out[9]  = (4.0L * in0 + 12.0L * in1 - 12.0L * in2 - 36.0L * in4 + 3.0L * in7 + 9.0L * in8 + 6.0L * in9 + 6.0L * in10 - 6.0L * in11 +
                           18.0L * in13 + 18.0L * in14 + 2.0L * in16 + 3.0L * in17 + 9.0L * in18) / 144.0L;
                out[10] = (4.0L * in0 - 12.0L * in1 + 12.0L * in2 - 36.0L * in4 + 3.0L * in7 + 9.0L * in8 + 6.0L * in9 - 6.0L * in10 + 6.0L * in11 -
                           18.0L * in13 - 18.0L * in14 + 2.0L * in16 + 3.0L * in17 + 9.0L * in18) / 144.0L;
                out[11] = (4.0L * in0 + 12.0L * in1 + 12.0L * in3 + 36.0L * in6 + 3.0L * in7 - 9.0L * in8 + 6.0L * in9 + 6.0L * in10 + 6.0L * in12 -
                           18.0L * in13 + 18.0L * in15 + 2.0L * in16 + 3.0L * in17 - 9.0L * in18) / 144.0L;
                out[12] = (4.0L * in0 - 12.0L * in1 - 12.0L * in3 + 36.0L * in6 + 3.0L * in7 - 9.0L * in8 + 6.0L * in9 - 6.0L * in10 - 6.0L * in12 +
                           18.0L * in13 - 18.0L * in15 + 2.0L * in16 + 3.0L * in17 - 9.0L * in18) / 144.0L;
                out[13] = (4.0L * in0 + 12.0L * in1 - 12.0L * in3 - 36.0L * in6 + 3.0L * in7 - 9.0L * in8 + 6.0L * in9 + 6.0L * in10 - 6.0L * in12 -
                           18.0L * in13 - 18.0L * in15 + 2.0L * in16 + 3.0L * in17 - 9.0L * in18) / 144.0L;
                out[14] = (4.0L * in0 - 12.0L * in1 + 12.0L * in3 - 36.0L * in6 + 3.0L * in7 - 9.0L * in8 + 6.0L * in9 - 6.0L * in10 + 6.0L * in12 +
                           18.0L * in13 + 18.0L * in15 + 2.0L * in16 + 3.0L * in17 - 9.0L * in18) / 144.0L;
                out[15] = (4.0L * in0 + 12.0L * in2 + 12.0L * in3 + 36.0L * in5 - 6.0L * in7 + 6.0L * in9 + 6.0L * in11 + 6.0L * in12 +
                           18.0L * in14 - 18.0L * in15 + 2.0L * in16 - 6.0L * in17) / 144.0L;
                out[16] = (4.0L * in0 - 12.0L * in2 - 12.0L * in3 + 36.0L * in5 - 6.0L * in7 + 6.0L * in9 - 6.0L * in11 - 6.0L * in12 -
                           18.0L * in14 + 18.0L * in15 + 2.0L * in16 - 6.0L * in17) / 144.0L;
                out[17] = (4.0L * in0 + 12.0L * in2 - 12.0L * in3 - 36.0L * in5 - 6.0L * in7 + 6.0L * in9 + 6.0L * in11 - 6.0L * in12 +
                           18.0L * in14 + 18.0L * in15 + 2.0L * in16 - 6.0L * in17) / 144.0L;
                out[18] = (4.0L * in0 - 12.0L * in2 + 12.0L * in3 - 36.0L * in5 - 6.0L * in7 + 6.0L * in9 - 6.0L * in11 + 6.0L * in12 -
                           18.0L * in14 - 18.0L * in15 + 2.0L * in16 - 6.0L * in17) / 144.0L;
            }
            














        private:
            InputData m_idata;
            size_t m_nVelocity;


            


        };








        // Equation::Equation(InputData idata)
        // : m_idata(idata), DataArray(idata){
    
        //     m_nVelocity = m_idata.nVelocity;
    
        // }
    
    
    
        // Equation::~Equation() {
        // }
    
    
    
        // __device__ __forceinline__ void Equation::convertD2Q9Weighted(const CLIP_REAL in[9], CLIP_REAL out[9]) {
        //     const CLIP_REAL in0 = in[0], in1 = in[1], in2 = in[2];
        //     const CLIP_REAL in3 = in[3], in4 = in[4], in5 = in[5];
        //     const CLIP_REAL in6 = in[6], in7 = in[7], in8 = in[8];
        
        //     out[0] = in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8;
        
        //     out[1] = -4.0 * in0 - in1 - in2 - in3 - in4 + 2.0 * in5 + 2.0 * in6 + 2.0 * in7 + 2.0 * in8;
        
        //     out[2] =  4.0 * in0 - 2.0 * in1 - 2.0 * in2 - 2.0 * in3 - 2.0 * in4 + in5 + in6 + in7 + in8;
        
        //     out[3] =  in1 - in3 + in5 - in6 - in7 + in8;
        
        //     out[4] = -2.0 * in1 + 2.0 * in3 + in5 - in6 - in7 + in8;
        
        //     out[5] =  in2 - in4 + in5 + in6 - in7 - in8;
        
        //     out[6] = -2.0 * in2 + 2.0 * in4 + in5 + in6 - in7 - in8;
        
        //     out[7] =  in1 - in2 + in3 - in4;
        
        //     out[8] =  in5 - in6 + in7 - in8;
        // }
    
    
        // __device__ __forceinline__ void Equation::reconvertD2Q9Weighted(const CLIP_REAL in[9], CLIP_REAL out[9]) {
        //     const CLIP_REAL in0 = in[0], in1 = in[1], in2 = in[2];
        //     const CLIP_REAL in3 = in[3], in4 = in[4], in5 = in[5];
        //     const CLIP_REAL in6 = in[6], in7 = in[7], in8 = in[8];
        
        //     out[0] = (4.0 * in0 - 4.0 * in1 + 4.0 * in2) / 36.0;
        
        //     out[1] = (4.0 * in0 - in1 - 2.0 * in2 + 6.0 * in3 - 6.0 * in4 + 9.0 * in7) / 36.0;
        
        //     out[2] = (4.0 * in0 - in1 - 2.0 * in2 + 6.0 * in5 - 6.0 * in6 - 9.0 * in7) / 36.0;
        
        //     out[3] = (4.0 * in0 - in1 - 2.0 * in2 - 6.0 * in3 + 6.0 * in4 + 9.0 * in7) / 36.0;
        
        //     out[4] = (4.0 * in0 - in1 - 2.0 * in2 - 6.0 * in5 + 6.0 * in6 - 9.0 * in7) / 36.0;
        
        //     out[5] = (4.0 * in0 + 2.0 * in1 + in2 + 6.0 * in3 + 3.0 * in4 + 6.0 * in5 + 3.0 * in6 + 9.0 * in8) / 36.0;
        
        //     out[6] = (4.0 * in0 + 2.0 * in1 + in2 - 6.0 * in3 - 3.0 * in4 + 6.0 * in5 + 3.0 * in6 - 9.0 * in8) / 36.0;
        
        //     out[7] = (4.0 * in0 + 2.0 * in1 + in2 - 6.0 * in3 - 3.0 * in4 - 6.0 * in5 - 3.0 * in6 + 9.0 * in8) / 36.0;
        
        //     out[8] = (4.0 * in0 + 2.0 * in1 + in2 + 6.0 * in3 + 3.0 * in4 - 6.0 * in5 - 3.0 * in6 - 9.0 * in8) / 36.0;
        // }
        


    
}



namespace equation {

    __device__ __forceinline__ void convertD2Q9Weighted(const CLIP_REAL in[9], CLIP_REAL out[9]) {
        const CLIP_REAL in0 = in[0], in1 = in[1], in2 = in[2];
        const CLIP_REAL in3 = in[3], in4 = in[4], in5 = in[5];
        const CLIP_REAL in6 = in[6], in7 = in[7], in8 = in[8];
    
        out[0] = in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8;
    
        out[1] = -4.0 * in0 - in1 - in2 - in3 - in4 + 2.0 * in5 + 2.0 * in6 + 2.0 * in7 + 2.0 * in8;
    
        out[2] =  4.0 * in0 - 2.0 * in1 - 2.0 * in2 - 2.0 * in3 - 2.0 * in4 + in5 + in6 + in7 + in8;
    
        out[3] =  in1 - in3 + in5 - in6 - in7 + in8;
    
        out[4] = -2.0 * in1 + 2.0 * in3 + in5 - in6 - in7 + in8;
    
        out[5] =  in2 - in4 + in5 + in6 - in7 - in8;
    
        out[6] = -2.0 * in2 + 2.0 * in4 + in5 + in6 - in7 - in8;
    
        out[7] =  in1 - in2 + in3 - in4;
    
        out[8] =  in5 - in6 + in7 - in8;
    }



    __device__ __forceinline__ void reconvertD2Q9Weighted(const CLIP_REAL in[9], CLIP_REAL out[9]) {
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
    

}


