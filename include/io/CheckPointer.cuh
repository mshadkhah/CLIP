// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file
 * @brief Defines the CheckPointer class for saving and restoring simulation state in the CLIP LBM framework.
 *
 * The CheckPointer class enables:
 * - Writing binary checkpoint files for key simulation variables
 * - Restoring a simulation from saved checkpoint state
 * - Saving and checking domain size and time step consistency
 * - Managing multiple rotated checkpoint folders
 *
 * This mechanism allows long-running simulations to be resumed or staged across runs.
 */

#pragma once
#include "includes.h"
#include "InputData.cuh"
#include "DataArray.cuh"
#include "Domain.cuh"
#include "TimeInfo.cuh"
#include "Boundary.cuh"

namespace clip
{

    /**
     * @brief Handles checkpointing for saving and resuming LBM simulations.
     *
     * The CheckPointer class provides functionality to save and load simulation state from binary files,
     * including all primary fields, time step information, and domain size. This enables resuming
     * long simulations without restarting from the beginning.
     */
    class CheckPointer
    {

    public:
        /**
         * @brief Constructs the CheckPointer and links required components.
         * @param DA Reference to data array manager
         * @param idata Reference to simulation input
         * @param domain Reference to simulation domain
         * @param ti Reference to time-tracking object
         * @param boundary Reference to boundary condition data
         */
        explicit CheckPointer(DataArray &DA, const InputData &idata, const Domain &domain, TimeInfo &ti, const Boundary &boundary);
        /// Virtual destructor
        virtual ~CheckPointer();

        /**
         * @brief Saves the current simulation state to a binary checkpoint folder.
         */
        void save();

        /**
         * @brief Loads simulation state from the latest checkpoint folder.
         */
        void load();

    private:
        const Domain *m_domain;     ///< Pointer to domain info
        const InputData *m_idata;   ///< Pointer to input config
        const Boundary *m_boundary; ///< Pointer to boundary info
        TimeInfo *m_ti;             ///< Pointer to time information
        DataArray *m_DA;            ///< Pointer to simulation fields

        std::string m_folder;   ///< Path to checkpoint folder
        std::string m_baseName; ///< Base file name for field data

        /**
         * @brief Saves host memory to binary file after copying from device.
         * @tparam T Data type
         * @param hostPtr Host-side pointer
         * @param devPtr Device-side pointer
         * @param ndof Number of degrees of freedom (default = scalar)
         * @param folder Target folder path
         * @param name Field name (used as file name)
         */
        template <typename T>
        void saveToFile(T *&hostPtr, const T *devPtr, CLIP_UINT ndof, const std::string &folder, const std::string &name);

        /**
         * @brief Loads data from binary file to host, then copies to device.
         * @tparam T Data type
         * @param devPtr Device pointer (output)
         * @param hostPtr Host pointer (temporary)
         * @param ndof Number of degrees of freedom
         * @param folder Checkpoint folder path
         * @param name Field name
         */
        template <typename T>
        void loadFromFile(T *&devPtr, T *&hostPtr, CLIP_UINT ndof, const std::string &folder, const std::string &name);

        /**
         * @brief Saves current time step and physical time to file.
         */
        void saveTimeInfo(const std::string &folder, const std::string &name);

        /**
         * @brief Loads time step and time from checkpoint file.
         */
        void loadTimeInfo(const std::string &folder, const std::string &name);

        /**
         * @brief Saves the domain size to file for consistency checking.
         */
        void saveDomainSize(const std::string &folder, const std::string &filename);

        /**
         * @brief Validates current domain size against saved one in checkpoint.
         */
        void checkDomainSize(const std::string &folder, const std::string &filename);

        /**
         * @brief Rotates folders to maintain a fixed number of checkpoint copies.
         * @param folder Root folder to rotate
         */
        void rotateFolders(const std::string &folder);

        /**
         * @brief Writes a text summary with time step and config info.
         */
        void saveSummaryInfo(const std::string &folder, const std::string &filename);
    };

}
