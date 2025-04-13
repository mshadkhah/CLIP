#pragma once

#include <includes.h>



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