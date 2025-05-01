# Copyright (c) 2020–2025 Mehdi Shadkhah
# SPDX-License-Identifier: BSD-3-Clause
# Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

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
  ${CMAKE_SOURCE_DIR}/include/io
  ${CMAKE_SOURCE_DIR}/include/geometry
)


set(CLIP_INC
  include/utils/includes.h
  include/utils/DataTypes.cuh
  include/data/DataArray.cuh
  include/utils/TimeInfo.cuh
  include/utils/Logger.cuh
  include/utils/Reporter.cuh
  include/input/InputData.cuh
  include/boundary/Boundary.cuh
  include/solver/Solver.cuh
  include/solver/WMRT.cuh
  include/equation/NsAllen.cuh
  include/domain/Domain.cuh
  include/io/VTSwriter.cuh
  include/io/CheckPointer.cuh
  include/geometry/Geometry.cuh
)


# Source files
set(CLIP_SRC
  src/data/DataArray.cu
  src/utils/TimeInfo.cu
  src/utils/Reporter.cu
  src/input/InputData.cu
  src/solver/Solver.cu
  src/equation/NsAllen.cu
  src/boundary/Boundary.cu
  src/domain/Domain.cu
  src/io/VTSwriter.cu
  src/io/CheckPointer.cu
  src/geometry/Geometry.cu
)



# Executable and its sources
add_executable(clip
  src/main.cu
)

add_library(clip_lib STATIC ${CLIP_SRC} ${CLIP_INC})



if (ENABLE_2D)
  set(ENABLE_3D OFF CACHE BOOL "Disabled 3D mode" FORCE)
elseif (ENABLE_3D)
  set(ENABLE_2D OFF CACHE BOOL "Disabled 2D mode" FORCE)
endif()


if (ENABLE_2D AND ENABLE_3D)
  message(FATAL_ERROR "You cannot enable both ENABLE_2D and ENABLE_3D.")
elseif (NOT ENABLE_2D AND NOT ENABLE_3D)
  message(STATUS "No mode specified — defaulting to 2D.")
  set(ENABLE_2D ON)
endif()


if (ENABLE_2D)
message(STATUS "Compiling in 2D mode.")
  target_compile_definitions(clip PRIVATE ENABLE_2D)
  target_compile_definitions(clip_lib PRIVATE ENABLE_2D)
elseif (ENABLE_3D)
message(STATUS "Compiling in 3D mode.")
  target_compile_definitions(clip PRIVATE ENABLE_3D)
  target_compile_definitions(clip_lib PRIVATE ENABLE_3D)
endif()




# Precision flag
option(USE_SINGLE_PRECISION "Use float instead of double" OFF)

if (USE_SINGLE_PRECISION)
    message(STATUS "Using single precision (float).")
    target_compile_definitions(clip PRIVATE USE_SINGLE_PRECISION)
    target_compile_definitions(clip_lib PRIVATE USE_SINGLE_PRECISION)
else()
    message(STATUS "Using double precision (default).")
endif()


# CUDA-specific options
target_compile_options(clip_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
set_target_properties(clip PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)


# Link libraries
target_link_libraries(clip PRIVATE clip_lib CUDA::cudart Threads::Threads)
