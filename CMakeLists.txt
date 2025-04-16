# Copyright (c) 2025
# Author: Mehdi Shadkhah

cmake_minimum_required(VERSION 3.18)
project(CLIP LANGUAGES CXX CUDA)

# Set standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Optional IntelliSense define for MSVC
if (MSVC)
  add_definitions(-D__CUDACC__)
endif()

# Flag options
option(ENABLE_2D "Enable 2D computation" ON)
option(ENABLE_3D "Enable 3D computation" OFF)
option(USE_SINGLE_PRECISION "Use float instead of double" OFF)

# Include directories
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/utils
  ${CMAKE_SOURCE_DIR}/include/input
  ${CMAKE_SOURCE_DIR}/include/equation
  ${CMAKE_SOURCE_DIR}/include/data
  ${CMAKE_SOURCE_DIR}/include/boundary
  ${CMAKE_SOURCE_DIR}/include/domain
  ${CMAKE_SOURCE_DIR}/include/solver
)


set(CLIP_INC
  include/utils/includes.h
  include/utils/DataTypes.cuh
  include/data/DataArray.cuh
  include/utils/TimeInfo.cuh
  include/input/InputData.cuh
  include/boundary/Boundary.cuh
  include/solver/Solver.cuh
  include/equation/NsAllen.cuh
  include/domain/Domain.cuh
)


# Source files
set(CLIP_SRC
  src/data/DataArray.cu
  src/utils/TimeInfo.cu
  src/input/InputData.cu
  src/solver/Solver.cu
  src/equation/NsAllen.cu
  src/boundary/Boundary.cu
  src/domain/Domain.cu
  src/test.cu
)


add_library(clip_lib STATIC ${CLIP_SRC} ${CLIP_INC})




# Executable and its sources
add_executable(clip
  src/main.cu
)


# Preprocessor definitions
if (ENABLE_2D)
  target_compile_definitions(clip PRIVATE ENABLE_2D)
  target_compile_definitions(clip_lib PRIVATE ENABLE_2D)
elseif (ENABLE_3D)
  target_compile_definitions(clip PRIVATE ENABLE_3D)
  target_compile_definitions(clip_lib PRIVATE ENABLE_3D)
endif()

if (USE_SINGLE_PRECISION)
  target_compile_definitions(clip PRIVATE USE_SINGLE_PRECISION)
endif()

# CUDA-specific options
target_compile_options(clip_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
set_target_properties(clip PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)


# Link libraries
target_link_libraries(clip PRIVATE clip_lib CUDA::cudart Threads::Threads)
