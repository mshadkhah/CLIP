#include <CheckPointer.cuh>

namespace clip
{

    CheckPointer::CheckPointer(DataArray &DA, const InputData &idata, const Domain &domain, TimeInfo &ti, const Boundary &boundary)
        : m_DA(&DA), m_idata(&idata), m_domain(&domain), m_ti(&ti), m_boundary(&boundary)
    {
    }

    CheckPointer::~CheckPointer() = default;

    void CheckPointer::save()
    {

        const CLIP_UINT Q = WMRT::WMRTvelSet::Q;
        const CLIP_UINT step = m_ti->getCurrentStep();

        if (m_idata->params.checkpointCopy != 0 && step % m_idata->params.checkpointInterval == 0)
        {

            rotateFolders("Checkpoint");
            saveSummaryInfo("Checkpoint", "info");
            // saveToFile(m_DA->hostDA.host_c, m_DA->deviceDA.dev_c, SCALAR_FIELD, "Checkpoint", "dev_c");
            // saveToFile(m_DA->hostDA.host_p, m_DA->deviceDA.dev_p, SCALAR_FIELD, "Checkpoint", "dev_p");
            // saveToFile(m_DA->hostDA.host_rho, m_DA->deviceDA.dev_rho, SCALAR_FIELD, "Checkpoint", "dev_rho");
            // saveToFile(m_DA->hostDA.host_vel, m_DA->deviceDA.dev_vel, DIM, "Checkpoint", "dev_vel");
            saveToFile(m_DA->hostDA.host_g_post, m_DA->deviceDA.dev_g_post, Q, "Checkpoint", "dev_g_post");
            // m_DA->writeHostToFile(m_DA->hostDA.host_g_post, "Checkpoint/tsttt.bin", Q * m_domain->domainSize);
            // double * testt;
            // m_DA->allocateOnHost(testt, "test", Q);

            // loadFromFile(m_DA->deviceDA.dev_g_post, testt, Q, "Checkpoint","tsttt");
            // // m_DA->readHostFromFile(testt, "Checkpoint/tsttt.bin", Q * m_domain->domainSize);
            // for (int i = 0; i < 50000; i++)
            // printf("data: %f \n", testt[i]);
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

    void CheckPointer::load()
    {
        const CLIP_UINT Q = WMRT::WMRTvelSet::Q;
        checkDomainSize("Checkpoint", "domainSize");
        // saveToFile(m_DA->deviceDA.dev_c, m_DA->hostDA.host_c, SCALAR_FIELD, "Checkpoint", "dev_c");
        // saveToFile(m_DA->deviceDA.dev_p, m_DA->hostDA.host_p, SCALAR_FIELD, "Checkpoint", "dev_p");
        // saveToFile(m_DA->deviceDA.dev_rho, m_DA->hostDA.host_rho, SCALAR_FIELD, "Checkpoint", "dev_rho");
        // saveToFile(m_DA->deviceDA.dev_vel, m_DA->hostDA.host_vel, DIM, "Checkpoint", "dev_vel");
        loadFromFile(m_DA->deviceDA.dev_g_post, m_DA->hostDA.host_g_post, Q, "Checkpoint", "dev_g_post");

        // double gg[77220];
        // double test[10000];
        // for (int i = 0; i < 10000; i++)
        //     test[i] = 567;
        // m_DA->writeHostToFile(&test[0], "Checkpoint/test.bin", 10000);

        // double test2[10000];
        // m_DA->readHostFromFile(&gg[0], "Checkpoint/dev_g_post.bin", 77220);
        // for (int i = 0; i < 50000; i++)
        // printf("data: %f \n", m_DA->hostDA.host_g_post[i]);

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

    template <typename T>
    void CheckPointer::loadFromFile(T *&devPtr, T *&hostPtr, CLIP_UINT ndof, const std::string &folder, const std::string &name)
    {
        std::string filename = folder + "/" + name + ".bin";
        m_DA->readHostFromFile(hostPtr, filename, ndof * m_domain->domainSize);
        m_DA->copyToDevice(devPtr, hostPtr, name, ndof);
    }

    template <typename T>
    void CheckPointer::saveToFile(T *&hostPtr, const T *devPtr, CLIP_UINT ndof, const std::string &folder, const std::string &name)
    {
        // Copy device to host first
        m_DA->copyFromDevice(hostPtr, devPtr, name, ndof);

        // Create new empty "checkpoint/" folder
        std::filesystem::create_directories(folder);

        // Now save the current data inside fresh "checkpoint/"
        std::string filename = folder + "/" + name + ".bin";

        m_DA->writeHostToFile(hostPtr, filename, ndof * m_domain->domainSize);
    }

    void CheckPointer::saveTimeInfo(const std::string &folder, const std::string &name)
    {

        std::filesystem::create_directories(folder);

        std::string filename = folder + "/" + name + ".bin";

        m_DA->writeHostToFile(&m_ti->getSimInfo(), filename, SCALAR_FIELD);
    }

    void CheckPointer::loadTimeInfo(const std::string &folder, const std::string &name)
    {

        std::string filename = folder + "/" + name + ".bin";
        TimeInfo::simInfo tempInfo;

        m_DA->readHostFromFile(&tempInfo, filename, SCALAR_FIELD);

        m_ti->getSimInfo().currentStep = tempInfo.currentStep;
        m_ti->getSimInfo().currentTime = tempInfo.currentTime;
    }

    void CheckPointer::saveDomainSize(const std::string &folder, const std::string &filename)
    {
        // Create the full path
        std::filesystem::create_directories(folder);
        std::string filepath = folder + "/" + filename + ".bin";

        m_DA->writeHostToFile(&m_idata->params.N[0], filepath, MAX_DIM);
    }

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

    void CheckPointer::saveSummaryInfo(const std::string &folder, const std::string &filename)
    {
        std::filesystem::create_directories(folder); // Make sure folder exists

        std::string filepath = folder + "/" + filename + ".txt";

        std::ofstream file(filepath);
        if (!file.is_open())
        {
            Logger::Error("Failed to open file for writing: " + filepath);
            return;
        }

        // Write basic simulation info
        file << "# Summary Info\n";
        file << "CurrentTime: " << m_ti->getCurrentTime() << "\n";
        file << "CurrentStep: " << m_ti->getCurrentStep() << "\n";

        // Write domain size
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
