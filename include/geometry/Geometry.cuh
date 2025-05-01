// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file
 * @brief Defines the Geometry class for managing embedded objects and generating signed distance functions (SDFs)
 *        used in the CLIP LBM simulation framework.
 *
 * The Geometry class provides:
 * - High-level geometry specification (circle, sphere, square, cube, perturbation)
 * - CPU-side structure for reading and storing geometries from input
 * - GPU-ready structure (`GeometryDevice`) for evaluating level sets
 * - Signed distance function (SDF) calculation in 2D and 3D
 *
 * This class enables complex boundary embedding and domain-aware initial conditions.
 */

#pragma once

#include "InputData.cuh"
#include "includes.h"
#include "Domain.cuh"
#include <string>

#define MAX_GEOMETRIES 16

namespace clip
{

    /**
     * @brief Represents one or more geometric objects for level-set-based boundary embedding.
     *
     * The Geometry class handles reading, storing, and evaluating geometric objects like spheres, cubes,
     * and perturbed interfaces. It generates a device-accessible structure used by the solver for computing
     * signed distance functions (SDF) to these shapes.
     */
    class Geometry
    {
    public:
        /**
         * @brief Constructs the Geometry manager using simulation input parameters.
         * @param idata Reference to input configuration
         */
        explicit Geometry(const InputData &idata);

        /// Destructor
        ~Geometry();

        /**
         * @brief Enumeration of supported geometric object types.
         */
        enum class Type
        {
            Circle = 0,
            Sphere = 1,
            Square = 2,
            Cube = 3,
            Perturbation = 4,
            Unknown = 5,
            MAX = 6
        };

        /**
         * @brief Represents a user-defined geometric object in host memory.
         */
        struct Entry
        {
            Type type = Type::Unknown;
            CLIP_REAL center[MAX_DIM] = {0.0, 0.0, 0.0};
            CLIP_REAL length[MAX_DIM] = {0.0, 0.0, 0.0};
            CLIP_REAL radius = 0.0;
            CLIP_REAL amplitude = 0.0;
            int id = -1;
        };

        /**
         * @brief Device-side flattened structure storing all geometry info used in kernels.
         */
        struct GeometryDevice
        {
            CLIP_INT numGeometries = 0;
            CLIP_INT type[MAX_GEOMETRIES];
            CLIP_REAL center[MAX_GEOMETRIES][MAX_DIM];
            CLIP_REAL length[MAX_GEOMETRIES][MAX_DIM];
            CLIP_REAL radius[MAX_GEOMETRIES];
            CLIP_REAL amplitude[MAX_GEOMETRIES];
            CLIP_INT id[MAX_GEOMETRIES];
        };

        /**
         * @brief Evaluates the signed distance function (SDF) to a given object at a spatial location.
         *
         * @param geo Reference to device geometry pool
         * @param id Geometry ID to match
         * @param x X-coordinate
         * @param y Y-coordinate
         * @param z Z-coordinate
         * @return Signed distance (positive = outside, negative = inside)
         */
        __device__ __host__ inline CLIP_REAL static sdf(const GeometryDevice &geo, CLIP_INT id, CLIP_REAL x, CLIP_REAL y, CLIP_REAL z)
        {
            for (int i = 0; i < geo.numGeometries; ++i)
            {
                if (geo.id[i] == id)
                {
                    switch (geo.type[i])
                    {
                    case static_cast<CLIP_INT>(Type::Circle):
                    {
                        CLIP_REAL dx = x - geo.center[i][IDX_X];
                        CLIP_REAL dy = y - geo.center[i][IDX_Y];
                        return sqrt(dx * dx + dy * dy) - geo.radius[i];
                    }
                    case static_cast<CLIP_INT>(Type::Sphere):
                    {
                        CLIP_REAL dx = x - geo.center[i][IDX_X];
                        CLIP_REAL dy = y - geo.center[i][IDX_Y];
                        CLIP_REAL dz = z - geo.center[i][IDX_Z];
                        return sqrt(dx * dx + dy * dy + dz * dz) - geo.radius[i];
                    }
                    case static_cast<CLIP_INT>(Type::Square):
                    {
                        CLIP_REAL dx = fabs(x - geo.center[i][IDX_X]) - geo.length[i][IDX_X] * 0.5;
                        CLIP_REAL dy = fabs(y - geo.center[i][IDX_Y]) - geo.length[i][IDX_Y] * 0.5;
                        CLIP_REAL ax = max(dx, 0.0);
                        CLIP_REAL ay = max(dy, 0.0);
                        CLIP_REAL outside = sqrt(ax * ax + ay * ay);
                        CLIP_REAL inside = min(max(dx, dy), 0.0);
                        return outside + inside;
                    }
                    case static_cast<CLIP_INT>(Type::Cube):
                    {
                        CLIP_REAL dx = fabs(x - geo.center[i][IDX_X]) - geo.length[i][IDX_X] * 0.5;
                        CLIP_REAL dy = fabs(y - geo.center[i][1]) - geo.length[i][IDX_Y] * 0.5;
                        CLIP_REAL dz = fabs(z - geo.center[i][2]) - geo.length[i][IDX_Z] * 0.5;
                        CLIP_REAL ax = max(dx, 0.0);
                        CLIP_REAL ay = max(dy, 0.0);
                        CLIP_REAL az = max(dz, 0.0);
                        CLIP_REAL outside = sqrt(ax * ax + ay * ay + az * az);
                        CLIP_REAL inside = min(max(max(dx, dy), dz), 0.0);
                        return outside + inside;
                    }
                    case static_cast<CLIP_INT>(Type::Perturbation):
                    {
#ifdef ENABLE_2D
                        const CLIP_REAL perturbation = geo.amplitude[i] * geo.length[i][IDX_X] * cos(2.0 * M_PI * x / geo.length[i][IDX_X]);

#elif defined(ENABLE_3D)
                        const CLIP_REAL perturbation = geo.amplitude[i] * geo.length[i][IDX_X] * (cos(2.0 * M_PI * x / geo.length[i][IDX_X]) + cos(2.0 * M_PI * z / geo.length[i][IDX_X]));
#endif
                        const CLIP_REAL yShift = perturbation + geo.center[i][IDX_Y];
                        return y - yShift;
                    }
                    }
                }
            }

            return 1e10; // if no matching object
        }

        /**
         * @brief Returns the device-side structure containing all geometry data.
         */
        const GeometryDevice &getDeviceStruct() const { return m_deviceGeometry; }

        /**
         * @brief Prints all geometry entries for debug or verification.
         */
        void print() const;

    private:
        const InputData *m_idata;        ///< Pointer to input parameters
        std::vector<Entry> geometries;   ///< Host-side geometry list
        GeometryDevice m_deviceGeometry; ///< GPU-ready geometry structure
        CLIP_UINT geometryObjects = 0;   ///< Number of objects parsed

        /**
         * @brief Reads geometry definitions from input and fills the entries vector.
         */
        bool readGeometries(std::vector<Entry> &geometries);

        /**
         * @brief Populates the GeometryDevice structure with data from entries.
         */
        void fillDeviceGeometry();

        /**
         * @brief Converts a string to lowercase (utility).
         */
        std::string toLower(const std::string &s);

        /**
         * @brief Trims whitespace from a string.
         */
        void trim(std::string &s);

        /**
         * @brief Parses a geometry type from a string.
         */
        Type typeFromString(const std::string &str);

        /**
         * @brief Converts a geometry type enum to string.
         */
        std::string typeToString(Type t) const;
    };

} // namespace clip
