#include <VTSwriter.cuh>

namespace clip
{

    VTSwriter::VTSwriter(DataArray &DA, const InputData &idata, const Domain &domain, const TimeInfo &ti, const std::string &folder, const std::string &baseName)
        : m_DA(&DA), m_idata(&idata), m_domain(&domain), m_ti(&ti), m_folder(folder), m_baseName(baseName)
    {
    }

    VTSwriter::~VTSwriter() = default;


    void VTSwriter::writeScalar(std::ofstream &file)
    {
        writeScalarArray(file, m_DA->hostDA.host_c, "C");
    }

    void VTSwriter::writeField(std::ofstream &file)
    {
        // writeScalarArray(file, m_DA->hostDA.host_vel, "Velocity");
    }



    void VTSwriter::writeToFile()
    {
        m_DA->updateHost();
        if(m_ti->getCurrentStep() % m_idata->params.outputInterval == 0){
            writeVTSBinaryFile();
        }
    }





  

    void VTSwriter::writeVTSBinaryFile()
    {
        // Create output directory if it doesn't exist
        std::filesystem::create_directory(m_folder);

        // Construct full path to the output file
        std::ostringstream filename;
        filename << m_folder << "/" << m_baseName << "_t" << std::fixed << std::setprecision(4) << m_ti->getCurrentStep() << ".vts";

        std::ofstream file(filename.str());

        if (!file.is_open())
        {
            Logger::Error("Failed to open file: " + filename.str());
            return;
        }

        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";

        file << "<StructuredGrid WholeExtent=\""
             << m_domain->info.domainMinIdx[IDX_X] << " " << m_domain->info.domainMaxIdx[IDX_X] << " "
             << m_domain->info.domainMinIdx[IDX_Y] << " " << m_domain->info.domainMaxIdx[IDX_Y] << " ";
#ifdef ENABLE_3D
        file << m_domain->info.domainMinIdx[IDX_Z] << " " << m_domain->info.domainMaxIdx[IDX_Z];
#else
        file << "0 0";
#endif
        file << "\">\n";

        file << "<Piece Extent=\""
             << m_domain->info.domainMinIdx[IDX_X] << " " << m_domain->info.domainMaxIdx[IDX_X] << " "
             << m_domain->info.domainMinIdx[IDX_Y] << " " << m_domain->info.domainMaxIdx[IDX_Y] << " ";
#ifdef ENABLE_3D
        file << m_domain->info.domainMinIdx[IDX_Z] << " " << m_domain->info.domainMaxIdx[IDX_Z];
#else
        file << "0 0";
#endif
        file << "\">\n";

        // Points section
        file << "<Points>\n<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (CLIP_UINT k = m_domain->info.domainMinIdx[IDX_Z]; k <= m_domain->info.domainMaxIdx[IDX_Z]; k++)
            for (CLIP_UINT j = m_domain->info.domainMinIdx[IDX_Y]; j <= m_domain->info.domainMaxIdx[IDX_Y]; j++)
                for (CLIP_UINT i = m_domain->info.domainMinIdx[IDX_X]; i <= m_domain->info.domainMaxIdx[IDX_X]; i++)
                    file << CLIP_REAL(i) << " " << CLIP_REAL(j) << " " << CLIP_REAL(k) << "\n";
        file << "</DataArray>\n</Points>\n";

        // // Scalar and vector data
        file << "<PointData Scalars=\"scalars\" Vectors=\"velocity\">\n";

        writeField(file);

        writeScalar(file);

        file << "</PointData>\n";
        file << "</Piece>\n</StructuredGrid>\n</VTKFile>\n";
        file.close();
    }

    void VTSwriter::writeScalarArray(std::ofstream &file, CLIP_REAL *data, const std::string &name)
    {

        file << "<DataArray type=\"Float64\" Name=\"" << name << "\" format=\"ascii\">\n";

        int counter = 0;
        for (CLIP_UINT k = m_domain->info.domainMinIdx[IDX_Z]; k <= m_domain->info.domainMaxIdx[IDX_Z]; k++)
        {
            for (CLIP_UINT j = m_domain->info.domainMinIdx[IDX_Y]; j <= m_domain->info.domainMaxIdx[IDX_Y]; j++)
            {
                for (CLIP_UINT i = m_domain->info.domainMinIdx[IDX_X]; i <= m_domain->info.domainMaxIdx[IDX_X]; i++)
                {
                    const CLIP_UINT idx_SCALAR = Domain::getIndex(m_domain->info, i, j, k);
                    file << data[idx_SCALAR] << "\n";
                    counter++;
                }
            }
        }
        std::cout << "count: " << counter << std::endl;

        file << "</DataArray>\n";
    }

    void VTSwriter::writeFieldArray(std::ofstream &file, CLIP_REAL *data, const std::string &name)
    {

        file << "<DataArray type=\"Float64\" Name=\"" << name
             << "\" NumberOfComponents=\"" << DIM
             << "\" format=\"ascii\">\n";

        for (CLIP_UINT k = m_domain->info.domainMinIdx[IDX_Z]; k <= m_domain->info.domainMaxIdx[IDX_Z]; k++)
        {
            for (CLIP_UINT j = m_domain->info.domainMinIdx[IDX_Y]; j <= m_domain->info.domainMaxIdx[IDX_Y]; j++)
            {
                for (CLIP_UINT i = m_domain->info.domainMinIdx[IDX_X]; i <= m_domain->info.domainMaxIdx[IDX_X]; i++)
                {

                    const CLIP_UINT idx_X = Domain::getIndex<DIM>(m_domain->info, i, j, k, IDX_X);
                    const CLIP_UINT idx_Y = Domain::getIndex<DIM>(m_domain->info, i, j, k, IDX_Y);

#ifdef ENABLE_3D
                    const CLIP_UINT idx_Z = Domain::getIndex<DIM>(m_domain->info, i, j, k, IDX_Z);
#endif

                    file << data[idx_X] << " " << data[idx_Y];
#ifdef ENABLE_3D
                    file << " " << data[idx_Z];
#endif
                    file << "\n";
                }
            }
        }

        file << "</DataArray>\n";
    }















}
