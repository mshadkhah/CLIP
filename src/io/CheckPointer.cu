// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file CheckPointer.cu
 * @brief Handles checkpointing functionality for saving and restoring LBM simulation state.
 *
 * @details
 * This module defines the `CheckPointer` class, responsible for:
 * - Saving the full simulation state (`f`, `g`, time, domain) at specified intervals
 * - Supporting multiple rotated checkpoint copies (e.g., Checkpoint_1, Checkpoint_2, ...)
 * - Restoring simulation state from checkpoint folders for continuation
 *
 * Stored components:
 * - Distribution functions: `f`, `g`, `f_post`, `g_post`, optionally `f_prev`, `g_prev`
 * - Time information (`currentStep`, `currentTime`)
 * - Domain size to ensure checkpoint compatibility
 * - Summary info for logging/resume validation
 *
 * @author
 * Mehdi Shadkhah
 *
 * @date
 * 2025
 */

#include <CheckPointer.cuh>

namespace clip
{

    /// Constructs the CheckPointer and links required components.
    /// @param DA Reference to data array manager
    /// @param idata Reference to simulation input
    /// @param domain Reference to simulation domain
    /// @param ti Reference to time-tracking object
    /// @param boundary Reference to boundary condition data

    CheckPointer::CheckPointer(DataArray &DA, const InputData &idata, const Domain &domain, TimeInfo &ti, const Boundary &boundary)
        : m_DA(&DA), m_idata(&idata), m_domain(&domain), m_ti(&ti), m_boundary(&boundary)
    {
    }

    /**
     * @brief Destructor for CheckPointer.
     */
    CheckPointer::~CheckPointer() = default;

    /**
     * @brief Saves the current simulation state to disk if checkpointing is enabled and interval is reached.
     */
    void CheckPointer::save()
    {

        const CLIP_UINT Q = WMRT::WMRTvelSet::Q;
        const CLIP_UINT step = m_ti->getCurrentStep();

        if (m_idata->params.checkpointCopy != 0 && step % m_idata->params.checkpointInterval == 0)
        {
            rotateFolders("Checkpoint");
            saveSummaryInfo("Checkpoint", "info");
            saveToFile(m_DA->hostDA.host_g_post, m_DA->deviceDA.dev_g_post, Q, "Checkpoint", "dev_g_post");
            saveToFile(m_DA->hostDA.host_f_post, m_DA->deviceDA.dev_f_post, Q, "Checkpoint", "dev_f_post");
            saveToFile(m_DA->hostDA.host_g, m_DA->deviceDA.dev_g, Q, "Checkpoint", "dev_g");
            saveToFile(m_DA->hostDA.host_f, m_DA->deviceDA.dev_f, Q, "Checkpoint", "dev_f");

            if (m_boundary->isFreeConvect)
            {
                saveToFile(m_DA->hostDA.host_g_prev, m_DA->deviceDA.dev_g_prev, Q, "Checkpoint", "dev_g_prev");
                saveToFile(m_DA->hostDA.host_f_prev, m_DA->deviceDA.dev_f_prev, Q, "Checkpoint", "dev_f_prev");
            }

            saveTimeInfo("Checkpoint", "timeInfo");
            saveDomainSize("Checkpoint", "domainSize");

            Logger::Success("Checkpoint successfully saved.");
        }
    }

    /**
     * @brief Loads a previously saved checkpoint from disk and restores simulation state.
     */
    void CheckPointer::load()
    {
        const CLIP_UINT Q = WMRT::WMRTvelSet::Q;
        checkDomainSize("Checkpoint", "domainSize");
        loadFromFile(m_DA->deviceDA.dev_g_post, m_DA->hostDA.host_g_post, Q, "Checkpoint", "dev_g_post");
        loadFromFile(m_DA->deviceDA.dev_f_post, m_DA->hostDA.host_f_post, Q, "Checkpoint", "dev_f_post");
        loadFromFile(m_DA->deviceDA.dev_g, m_DA->hostDA.host_g, Q, "Checkpoint", "dev_g");
        loadFromFile(m_DA->deviceDA.dev_f, m_DA->hostDA.host_f, Q, "Checkpoint", "dev_f");

        if (m_boundary->isFreeConvect)
        {
            loadFromFile(m_DA->deviceDA.dev_g_prev, m_DA->hostDA.host_g_prev, Q, "Checkpoint", "dev_g_prev");
            loadFromFile(m_DA->deviceDA.dev_f_prev, m_DA->hostDA.host_f_prev, Q, "Checkpoint", "dev_f_prev");
        }

        loadTimeInfo("Checkpoint", "timeInfo");

        Logger::Success("Checkpoint successfully loaded.");
    }

    /**
     * @brief Rotates checkpoint folders to maintain a fixed number of saved copies.
     * @param folder Base folder name for checkpoints (e.g., "Checkpoint")
     */
    void CheckPointer::rotateFolders(const std::string &folder)
    {
        if (std::filesystem::exists(folder))
        {
            for (int i = m_idata->params.checkpointCopy - 1; i >= 1; --i)
            {
                std::string oldFolder = folder + "_" + std::to_string(i);
                std::string newFolder = folder + "_" + std::to_string(i + 1);

                if (std::filesystem::exists(oldFolder))
                {
                    // If target exists, remove it first
                    if (std::filesystem::exists(newFolder))
                    {
                        std::filesystem::remove_all(newFolder);
                    }

                    std::filesystem::rename(oldFolder, newFolder);
                }
            }

            // Move current checkpoint to checkpoint_1
            std::string checkpoint1 = folder + "_1";
            if (std::filesystem::exists(checkpoint1))
            {
                std::filesystem::remove_all(checkpoint1);
            }

            std::filesystem::rename(folder, checkpoint1);
        }
    }

    /**
     * @brief Loads a field from file into device memory via host.
     * @tparam T Data type
     * @param devPtr Device pointer to load into
     * @param hostPtr Temporary host pointer for reading
     * @param ndof Number of degrees of freedom (Q)
     * @param folder Checkpoint folder path
     * @param name Field name (used for filename construction)
     */
    template <typename T>
    void CheckPointer::loadFromFile(T *&devPtr, T *&hostPtr, CLIP_UINT ndof, const std::string &folder, const std::string &name)
    {
        std::string filename = folder + "/" + name + ".bin";
        m_DA->readHostFromFile(hostPtr, filename, ndof * m_domain->domainSize);
        m_DA->copyToDevice(devPtr, hostPtr, name, ndof);
    }

    /**
     * @brief Saves a field from device to disk through host memory.
     * @tparam T Data type
     * @param hostPtr Temporary host pointer
     * @param devPtr Device pointer to read from
     * @param ndof Number of degrees of freedom (Q)
     * @param folder Checkpoint folder path
     * @param name Field name (used for filename construction)
     */
    template <typename T>
    void CheckPointer::saveToFile(T *&hostPtr, const T *devPtr, CLIP_UINT ndof, const std::string &folder, const std::string &name)
    {

        m_DA->copyFromDevice(hostPtr, devPtr, name, ndof);

        std::filesystem::create_directories(folder);

        std::string filename = folder + "/" + name + ".bin";

        m_DA->writeHostToFile(hostPtr, filename, ndof * m_domain->domainSize);
    }

    /**
     * @brief Saves the current time step and simulation time to file.
     * @param folder Destination folder
     * @param name Filename (without extension)
     */
    void CheckPointer::saveTimeInfo(const std::string &folder, const std::string &name)
    {

        std::filesystem::create_directories(folder);

        std::string filename = folder + "/" + name + ".bin";

        m_DA->writeHostToFile(&m_ti->getSimInfo(), filename, SCALAR_FIELD);
    }

    /**
     * @brief Loads simulation time information (step and time) from file.
     * @param folder Source folder
     * @param name Filename (without extension)
     */
    void CheckPointer::loadTimeInfo(const std::string &folder, const std::string &name)
    {

        std::string filename = folder + "/" + name + ".bin";
        TimeInfo::simInfo tempInfo;

        m_DA->readHostFromFile(&tempInfo, filename, SCALAR_FIELD);

        m_ti->getSimInfo().currentStep = tempInfo.currentStep;
        m_ti->getSimInfo().currentTime = tempInfo.currentTime;
    }

    /**
     * @brief Saves domain size (`N`) to a file to validate future checkpoint compatibility.
     * @param folder Destination folder
     * @param filename Filename (without extension)
     */
    void CheckPointer::saveDomainSize(const std::string &folder, const std::string &filename)
    {

        std::filesystem::create_directories(folder);
        std::string filepath = folder + "/" + filename + ".bin";

        m_DA->writeHostToFile(&m_idata->params.N[0], filepath, MAX_DIM);
    }

    /**
     * @brief Checks that the loaded domain size matches the currently configured simulation domain.
     * @param folder Checkpoint folder path
     * @param filename Filename to read domain size from
     */
    void CheckPointer::checkDomainSize(const std::string &folder, const std::string &filename)
    {
        std::string filepath = folder + "/" + filename + ".bin";

        CLIP_REAL N[MAX_DIM];

        m_DA->readHostFromFile(&N[0], filepath, MAX_DIM);

        for (size_t d = 0; d < MAX_DIM; ++d)
        {
            if (N[d] != m_idata->params.N[d])
            {
                Logger::Error("Loaded domain size mismatch at dimension " + std::to_string(d) +
                              ". Expected: " + std::to_string(m_idata->params.N[d]) +
                              ", Found: " + std::to_string(N[d]));
                return;
            }
        }
    }

    /**
     * @brief Writes summary information (time, step, domain size) into a human-readable `.txt` file.
     * @param folder Destination folder
     * @param filename Output filename (without extension)
     */
    void CheckPointer::saveSummaryInfo(const std::string &folder, const std::string &filename)
    {
        std::filesystem::create_directories(folder);

        std::string filepath = folder + "/" + filename + ".txt";

        std::ofstream file(filepath);
        if (!file.is_open())
        {
            Logger::Error("Failed to open file for writing: " + filepath);
            return;
        }

        file << "# Summary Info\n";
        file << "CurrentTime: " << m_ti->getCurrentTime() << "\n";
        file << "CurrentStep: " << m_ti->getCurrentStep() << "\n";

        file << "DomainSize (N): ";
        for (size_t d = 0; d < MAX_DIM; ++d)
        {
            file << m_idata->params.N[d];
            if (d < MAX_DIM - 1)
                file << ", ";
        }
        file << "\n";

        file.close();
    }

}
