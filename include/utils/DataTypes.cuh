#pragma once

#include <includes.h>



typedef unsigned int CLIP_UINT;
typedef int CLIP_INT;
#ifdef USE_SINGLE_PRECISION
typedef float CLIP_REAL;
#else
typedef double CLIP_REAL;
#endif