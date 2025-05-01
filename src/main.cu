// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file main.cpp
 * @brief Main entry point for CLIP: CUDA Lattice Boltzmann Solver for Interfacial Phenomena.
 *
 * @details
 * This file initializes all components of the CLIP framework, including:
 * - Reading input configuration
 * - Setting up domain, geometry, boundary conditions, and data arrays
 * - Initializing the solver and I/O utilities (e.g., reporters, checkpointing, output)
 * - Executing the main simulation loop
 * 
 * It supports both fresh runs and resuming from checkpoint using the `-resume` flag.
 *
 * Example usage:
 * @code
 * ./CLIP_simulation                // fresh run
 * ./CLIP_simulation -resume       // resume from last checkpoint
 * @endcode
 *
 * @author Mehdi Shadkhah
 * @date 2025
 */



#include <InputData.cuh>
#include "includes.h"
#include <TimeInfo.cuh>
#include <NsAllen.cuh>
#include <Boundary.cuh>
#include <DataArray.cuh>
#include "VTSwriter.cuh"
#include "Reporter.cuh"
#include "CheckPointer.cuh"
#include "Geometry.cuh"

int main(int argc, char *argv[])
{
    bool resumeFromCheckpoint = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "-resume")
        {
            resumeFromCheckpoint = true;
            clip::Logger::Info("Resume mode activated via command line.");
        }
    }

    clip::InputData input("/home/mehdi/projects/CLIP/examples/3D/Jet/config.txt");
    
    // clip::InputData input("config.txt");

    clip::Domain domain(input);
    clip::Geometry geom(input);
    clip::Boundary boundary(input, domain);
    clip::DataArray DA(input, domain, boundary);
    DA.createVectors();

    

    // if (Geometry::sdf(geom, 0, x, y, z) <= 0)
    // printf("vel: %f \n",clip::Geometry::sdf(geom.getDeviceStruct(), 0, 32, 128, 32));

    clip::NSAllen eqn(input, domain, DA, boundary, geom);
    clip::TimeInfo ti(input);
    clip::VTSwriter output(DA, input, domain, ti, "test", "test");
    clip::Reporter report(DA, input, domain, ti);
    clip::CheckPointer chechpoint(DA, input, domain, ti, boundary);

    if (resumeFromCheckpoint)
    {
        chechpoint.load();
        eqn.macroscopic();
    }
    else
    {
        eqn.initialCondition();
        DA.updateDevice();
        eqn.deviceInitializer();
    }


    output.writeToFile();
  

    while (ti.getCurrentStep() < ti.getFinalStep())
    {
        eqn.solve();
        ti.increment();

        output.writeToFile();
        report.print();   
        chechpoint.save();
    }

    return 0;
}
