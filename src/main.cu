// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file main.cu
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
 * ./Clip                // fresh run
 * ./Clip -resume       // resume from last checkpoint
 * @endcode
 *
 * @author Mehdi Shadkhah
 * @date 2025
 */

#include "InputData.cuh"
#include "includes.h"
#include "TimeInfo.cuh"
#include "NsAllen.cuh"
#include "Boundary.cuh"
#include "DataArray.cuh"
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

    clip::InputData inputData("config.txt");

    clip::Domain simDomain(inputData);
    clip::Geometry geometry(inputData);
    clip::Boundary boundaryConditions(inputData, simDomain);
    clip::DataArray dataArray(inputData, simDomain, boundaryConditions);
    dataArray.createVectors();

    clip::NSAllen nsSolver(inputData, simDomain, dataArray, boundaryConditions, geometry);
    clip::TimeInfo timeInfo(inputData);
    clip::VTSwriter vtsWriter(dataArray, inputData, simDomain, timeInfo, "results", "results");
    clip::Reporter reporter(dataArray, inputData, simDomain, timeInfo);
    clip::CheckPointer checkpointer(dataArray, inputData, simDomain, timeInfo, boundaryConditions);

    if (resumeFromCheckpoint)
    {
        checkpointer.load();
        nsSolver.macroscopic();
    }
    else
    {
        nsSolver.initialCondition();
        dataArray.updateDevice();
        nsSolver.deviceInitializer();
    }

    vtsWriter.writeToFile();

    while (timeInfo.getCurrentStep() < timeInfo.getFinalStep())
    {
        nsSolver.solve();
        timeInfo.increment();

        vtsWriter.writeToFile();
        reporter.print();
        checkpointer.save();
    }

    return 0;
}