// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena



/**
 * @file
 * @brief Defines the VTSwriter class, which writes CLIP simulation output in VTK Structured Grid (.vts) format.
 *
 * This writer supports:
 * - Scalar and vector field output (e.g., pressure, velocity)
 * - Binary file generation for use in ParaView or VisIt
 * - Organized time-series output with naming based on timestep
 *
 * It uses DataArray to extract simulation fields and writes them according to the VTK XML format for structured grids.
 */


#pragma once
#include "includes.h"
#include "InputData.cuh"
#include "DataArray.cuh"
#include "Domain.cuh"
#include "TimeInfo.cuh"

namespace clip
{


    /**
 * @brief Outputs simulation data in VTK Structured Grid (.vts) binary format for visualization.
 *
 * The VTSwriter class is responsible for exporting field data from CLIP simulations into `.vts` files
 * that can be visualized using tools like ParaView. It supports scalar fields (e.g., pressure, phase)
 * and vector fields (e.g., velocity) in binary format and organizes outputs using step and time metadata.
 */
    class VTSwriter
    {

    public:

        /**
     * @brief Constructs the writer with simulation metadata and output configuration.
     * @param DA Reference to data array manager
     * @param idata Input configuration object
     * @param domain Domain info
     * @param ti Time tracker
     * @param folder Output directory
     * @param baseName Base filename for output files
     */
        explicit VTSwriter(DataArray &DA, const InputData &idata, const Domain &domain, const TimeInfo &ti, const std::string &folder, const std::string &baseName);

            /// Destructor
        virtual ~VTSwriter();

        
    /**
     * @brief Writes the current state of the simulation to a .vts file.
     */
        void writeToFile();

        private:
        const Domain* m_domain;       ///< Pointer to domain object
        const InputData* m_idata;     ///< Pointer to input parameters
        const TimeInfo* m_ti;         ///< Pointer to time-tracking object
        DataArray* m_DA;              ///< Pointer to simulation data arrays
        std::string m_folder;         ///< Output folder path
        std::string m_baseName;       ///< Base name for generated files

 /**
     * @brief Writes scalar fields such as pressure or phase indicator to the .vts file.
     * @param file Output file stream
     */
    void writeScalar(std::ofstream &file);

    /**
     * @brief Writes vector fields such as velocity to the .vts file.
     * @param file Output file stream
     */
    void writeField(std::ofstream &file);

    /**
     * @brief Generic writer for N-component vector fields.
     * @param file Output stream
     * @param data Pointer to field data
     * @param name Name of the field
     */
    void writeFieldArray(std::ofstream &file, CLIP_REAL *data, const std::string &name);

    /**
     * @brief Generic writer for scalar fields.
     * @param file Output stream
     * @param data Pointer to scalar field data
     * @param name Name of the field
     */
    void writeScalarArray(std::ofstream &file, CLIP_REAL *data, const std::string &name);

    /**
     * @brief Low-level writer that formats and writes the .vts binary file.
     */
    void writeVTSBinaryFile();
    };

}
