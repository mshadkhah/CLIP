// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file TimeInfo.cu
 * @brief Handles simulation time tracking in CLIP.
 *
 * @details
 * The `TimeInfo` class manages simulation step counting and physical time tracking.
 * It stores time step information and allows incrementing and querying simulation state.
 * It also provides access to the internal `simInfo` structure used for checkpointing.
 *
 * This is essential for controlling simulation duration, reporting, and saving/restoring state.
 *
 * @author Mehdi Shadkhah
 * @date 2025
 */

#include <TimeInfo.cuh>

namespace clip
{

    /// Constructor for TimeInfo.
    /// idata InputData object containing simulation parameters.

    TimeInfo::TimeInfo(InputData idata)
        : m_idata(idata) {}

    /**
     * @brief Returns the current simulation step.
     * @return Current step as an integer.
     */

    int TimeInfo::getCurrentStep() const
    {
        return m_info.currentStep;
    }

    /**
     * @brief Returns the current simulation time based on time step and step count.
     * @return Current physical time.
     */

    double TimeInfo::getCurrentTime() const
    {
        return m_info.currentStep * m_info.dt;
    }

    /// Sets the simulation time step (dt).
    /// dt The desired time step.

    void TimeInfo::setTimeStep(double dt)
    {
        m_info.dtIsSet = true;
        m_info.dt = dt;
    }

    /**
     * @brief Increments the current simulation step counter by one.
     */

    void TimeInfo::increment()
    {
        m_info.currentStep++;
    }

    /**
     * @brief Returns the user-specified final step from the input configuration.
     * @return Final step as an integer.
     */

    double TimeInfo::getFinalStep() const
    {
        return m_idata.params.finalStep;
    }

    /**
     * @brief Returns a reference to the internal simInfo structure.
     * @return Reference to mutable simInfo.
     */

    TimeInfo::simInfo &clip::TimeInfo::getSimInfo()
    {
        return m_info;
    }

    /**
     * @brief Returns a const reference to the internal simInfo structure.
     * @return Const reference to simInfo.
     */

    const TimeInfo::simInfo &clip::TimeInfo::getSimInfo() const
    {
        return m_info;
    }

}
