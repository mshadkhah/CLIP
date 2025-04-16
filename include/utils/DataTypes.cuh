#pragma once

// #include <includes.h>



typedef unsigned int CLIP_UINT;
typedef int CLIP_INT;
#ifdef USE_SINGLE_PRECISION
typedef float CLIP_REAL;
#else
typedef double CLIP_REAL;
#endif


#ifdef ENABLE_2D
#define DIM 2
#elif defined(ENABLE_3D)
#define DIM 3
#endif

#define MAX_DIM 3

#define SCALAR 0
#define IDX_X 0
#define IDX_Y 1
#define IDX_Z 2



