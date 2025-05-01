// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena


/**
 * @file
 * @brief Defines the InputData class responsible for reading and storing simulation parameters
 *        in the CLIP LBM framework.
 *
 * This class reads configuration files (e.g., `config.txt`) and stores user-defined simulation
 * parameters in a structured format (SimParams). These include:
 * - Time control and output intervals
 * - Physical parameters (Reynolds, Bond, Capillary numbers, etc.)
 * - Domain geometry and gravity
 * - Phase field and interface properties
 * - Case type selection (e.g., bubble, drop, jet, RTI)
 *
 * All parameters are accessible via the public `params` field.
 */




#pragma once
#include "includes.h"

namespace clip
{

    /**
 * @brief Reads simulation parameters from a configuration file and stores them in a structured format.
 *
 * The InputData class parses a plain text config file and populates all required physical and numerical
 * parameters for the CLIP simulation. It supports scalar values, arrays, enums, and vector fields.
 */
    class InputData
    {
    public:

        /**
     * @brief Constructs the InputData object and optionally sets the config file path.
     * @param filename Configuration file (default: "config.txt")
     */
        explicit InputData(const std::string &filename = "config.txt");

         /// Parses the configuration file and fills the SimParams structure.
        void read_config();

        /// Returns the filename used for the configuration file.
        std::string getConfig() const;

            /**
     * @brief Enumerates supported simulation case types.
     */
    enum CaseType
    {
        Bubble = 0,   ///< Rising or stationary bubble
        Drop = 1,     ///< Falling or stationary drop
        RTI = 2,      ///< Rayleigh–Taylor instability
        Jet = 3,      ///< Liquid jet simulation
        Unknown = 4   ///< Unknown or unspecified case
    };


        /**
     * @brief Struct holding all simulation parameters, parsed from input.
     */
        struct SimParams
        {
            // Time and output control
            CLIP_REAL tFinal;
            CLIP_UINT finalStep;
            CLIP_UINT outputInterval;
            CLIP_UINT reportInterval;
            CLIP_UINT checkpointInterval = 0;
            CLIP_UINT checkpointCopy = 0;
       

            CLIP_REAL N[MAX_DIM];
            CLIP_REAL referenceLength;
            CLIP_REAL referenceVelocity;

            CLIP_REAL Bo;
            CLIP_REAL Re;
            CLIP_REAL We;
            CLIP_REAL Pe;
            CLIP_REAL Ca;
            CLIP_REAL Mo;

            CLIP_REAL rhoRatio;
            CLIP_REAL muRatio;

            CLIP_REAL gravity[MAX_DIM];
            CLIP_REAL interfaceWidth;
            CLIP_REAL mobility;

            CaseType caseType;

            CLIP_REAL RhoH;
            CLIP_REAL RhoL;

            CLIP_REAL sigma;

            CLIP_REAL radius;
            CLIP_REAL betaConstant;
            CLIP_REAL kConstant;

            CLIP_REAL tauL;
            CLIP_REAL tauH;

            CLIP_REAL muL;
            CLIP_REAL muH;

            CLIP_REAL amplitude;
        };

        /// Public simulation parameters populated from the config file
        SimParams params;

        private:
        std::string m_filename; ///< Path to the config file
    
        /// Generic read function for scalar variables
        template <typename T>
        bool read_value(const std::string &varName, T &var) const;
    
        /// Reads fixed-size array variables from config
        template <typename T, std::size_t N>
        bool read_array(const std::string& varName, T (&arr)[N]) const;
    
        /// Reads vectors of values from config
        template <typename T>
        bool read_vector(const std::string &varName, std::vector<T> &arr) const;
    
        /// Trims whitespace from both ends of a string
        static void trim(std::string &s);
    
        /// Overloaded read functions for specific types
        bool read(const std::string &varName, CLIP_REAL &var) const;
        bool read(const std::string &varName, CLIP_UINT &var) const;
        bool read(const std::string &varName, bool &var) const;
        template <typename T, std::size_t N>
        bool read(const std::string &varName, T (&arr)[N]) const;
        bool read(const std::string &varName, CaseType &caseType) const;
    
        /// Converts a string to CaseType enum
        static CaseType caseTypeFromString(const std::string &s);
    };

}
