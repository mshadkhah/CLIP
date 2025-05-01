// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena


/**
 * @file
 * @brief Defines the Boundary class for handling physical and computational boundary conditions 
 *        in the CLIP LBM simulation framework.
 *
 * This file includes:
 * - Boundary condition types (wall, velocity, periodic, etc.)
 * - Boundary side definitions (X±, Y±, Z±)
 * - Structures for organizing boundary condition data
 * - Device-accessible utilities for boundary type checks
 *
 * The Boundary class parses configuration input, initializes boundary metadata,
 * and exposes BC flags used throughout CLIP's solver kernels.
 */



#pragma once
#include "InputData.cuh"
#include "includes.h"
#include "Domain.cuh"

/**
 * @brief Boundary object identifiers used across the simulation domain.
 */
namespace object
{
    constexpr int XMinus = 0;
    constexpr int XPlus = 1;
    constexpr int YMinus = 2;
    constexpr int YPlus = 3;
    constexpr int ZMinus = 4;
    constexpr int ZPlus = 5;
    constexpr int Unknown = 6;
}

namespace clip
{

    /**
     * @brief Class for handling boundary condition definitions and utilities in a simulation domain.
     */
    class Boundary
    {
    public:
        /**
         * @brief Constructs the Boundary object and initializes boundaries from input data.
         * @param idata Reference to simulation input parameters.
         * @param domain Reference to the simulation domain.
         */
        explicit Boundary(const InputData &idata, const Domain &domain);

        /**
         * @brief Destructor for the Boundary class.
         */
        ~Boundary();

        /**
         * @brief Enum for identifying the six sides of a 3D domain.
         */
        enum class Objects
        {
            XMinus = 0,
            XPlus = 1,
            YMinus = 2,
            YPlus = 3,
            ZMinus = 4,
            ZPlus = 5,
            Unknown = 6,
            MAX = 7
        };

        /**
         * @brief Enum for different types of boundary conditions.
         */
        enum class Type
        {
            Wall = 0,
            SlipWall = 1,
            FreeConvect = 2,
            Periodic = 3,
            Neumann = 4,
            Velocity = 5,
            DoNothing = 6,
            Unknown = 7,
            MAX = 8
        };

        /**
         * @brief Represents a single boundary condition entry for a specific domain side.
         */

        struct Entry
        {
            Objects side = Objects::Unknown;
            Type BCtype = Type::Unknown;
            CLIP_REAL value[MAX_DIM];
            bool ifRefine = false;
        };

        /**
         * @brief Struct used for device-side access to all boundary types and values.
         */
        struct BCTypeMap
        {
            Boundary::Type types[static_cast<int>(Boundary::Objects::MAX)];
            CLIP_REAL val[static_cast<int>(Boundary::Objects::MAX)][MAX_DIM];
        };

        /**
         * @brief Determines whether the given boundary type is a mirror-type condition.
         * @param type Boundary condition type to test.
         * @return true if it is a mirror-type condition (e.g., wall, slip wall, etc.)
         */
        __device__ __forceinline__ static bool isMirrorType(Boundary::Type type)
        {
            return (type == Boundary::Type::Wall ||
                    type == Boundary::Type::FreeConvect ||
                    type == Boundary::Type::Neumann ||
                    type == Boundary::Type::SlipWall ||
                    type == Boundary::Type::DoNothing);
        }

        BCTypeMap BCMap;            ///< Device-accessible map of BC types and values
        bool isPeriodic = false;    ///< True if any periodic boundary is present
        bool isWall = false;        ///< True if any wall boundary is present
        bool isSlipWall = false;    ///< True if any slip wall boundary is present
        bool isFreeConvect = false; ///< True if any free convective boundary is present
        bool isNeumann = false;     ///< True if any Neumann boundary is present
        bool isVelocity = false;    ///< True if any velocity BC is present

    private:
        const InputData *m_idata; ///< Pointer to simulation input
        const Domain *m_domain;   ///< Pointer to the simulation domain

        dim3 dimBlock, dimGrid; ///< CUDA kernel launch configurations

        std::vector<Entry> boundaries; ///< List of parsed boundaries
        CLIP_UINT boundaryObjects;     ///< Total number of boundary sides in use

        /**
         * @brief Reads boundary definitions from the input and fills the vector.
         */
        bool readBoundaries(std::vector<Entry> &boundaries);

        /**
         * @brief Prints debug info about boundary configurations.
         */
        void print();

        /**
         * @brief Converts Type enum to its string representation.
         */
        std::string toString(Type type);

        /**
         * @brief Converts Objects enum to its string representation.
         */
        std::string toString(Objects side);

        /**
         * @brief Parses a boundary type from its string representation.
         */
        Type typeFromString(const std::string &str);

        /**
         * @brief Parses a domain side from its string representation.
         */
        Objects sideFromString(const std::string &str);

        /**
         * @brief Trims leading/trailing whitespace from a string.
         */
        void trim(std::string &s);

        /**
         * @brief Updates internal flags for fast boundary checks.
         */
        void updateFlags();

        /**
         * @brief Converts a string to lowercase.
         */
        std::string toLower(const std::string &s);
    };

} // namespace clip
