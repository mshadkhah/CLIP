// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file
 * @brief Declares the TimeInfo class, which manages simulation time and step tracking
 *        in the CLIP LBM framework.
 *
 * This utility provides:
 * - Current time and time step tracking
 * - Final time and iteration limits
 * - Interface for updating and accessing simulation time state
 *
 * It supports both time-based and step-based termination criteria.
 */

#pragma once
#include <includes.h>
#include <InputData.cuh>

namespace clip
{

    /**
     * @brief Handles time step tracking and advancement for LBM simulations.
     *
     * The TimeInfo class tracks the simulation clock, time step, and step count,
     * and provides an interface for advancing or querying the simulation state.
     */
    class TimeInfo
    {
    public:
        /**
         * @brief Holds simulation time-related quantities.
         */
        struct simInfo
        {
            double currentTime = 0; ///< Current simulation time
            int currentStep = 0;    ///< Current time step index
            double dt = 0;          ///< Time step size
            bool dtIsSet = false;   ///< Flag to indicate if `dt` was explicitly set
        };

        /**
         * @brief Constructs a TimeInfo manager using simulation input parameters.
         * @param idata The input configuration object
         */
        explicit TimeInfo(InputData idata);

        /**
         * @brief Returns the current time step index.
         */
        int getCurrentStep() const;

        /**
         * @brief Returns the current simulation time.
         */
        double getCurrentTime() const;

        /**
         * @brief Returns the final simulation time (`tFinal`) from input.
         */
        double getEndTime() const;

        /**
         * @brief Returns the maximum number of time steps (`finalStep`) from input.
         */
        double getFinalStep() const;

        /**
         * @brief Advances the simulation by one time step.
         */
        void increment();

        /**
         * @brief Sets the time step value.
         * @param dt The time step size
         */
        void setTimeStep(double dt);

        /**
         * @brief Returns a reference to the internal `simInfo` struct.
         */
        simInfo &getSimInfo();

        /**
         * @brief Returns a const reference to the internal `simInfo` struct.
         */
        const simInfo &getSimInfo() const;

    private:
        InputData m_idata; ///< Copy of simulation input for accessing time limits
        simInfo m_info;    ///< Current simulation time/step data
    };

} // namespace clip
