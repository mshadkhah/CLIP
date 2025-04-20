#include <VTSwriter.cuh>

namespace clip
{

    VTSwriter::VTSwriter(const InputData& idata, const Domain& domain, const TimeInfo& ti)
        : m_idata(&idata), m_domain(&domain), m_ti(&ti)
    {
    }

    // void WriteVTKBinaryFile(double t, int Nx, int Ny, double L0,
    //                         const double *host_ux, const double *host_uy,
    //                         const double *host_c, const double *host_p,
    //                         const double *host_rho)
    // {
    //     std::filesystem::create_directory("results");

    //     std::ostringstream filename;
    //     filename << "results/time_step_" << std::fixed << std::setprecision(4) << t << ".vts";
    //     std::ofstream file(filename.str());

    //     file << "<?xml version=\"1.0\"?>\n";
    //     file << "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    //     file << "<StructuredGrid WholeExtent=\"0 " << Nx << " 0 " << Ny << " 0 0\">\n";
    //     file << "<Piece Extent=\"0 " << Nx << " 0 " << Ny << " 0 0\">\n";

    //     file << "<Points>\n<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    //     for (int j = 0; j <= Ny; ++j)
    //         for (int i = 0; i <= Nx; ++i)
    //             file << double(i) / L0 << " " << double(j) / L0 << " 0.0\n";
    //     file << "</DataArray>\n</Points>\n";

    //     file << "<PointData Scalars=\"scalars\">\n";
    //     auto write_array = [&](const char *name, const double *data)
    //     {
    //         file << "<DataArray type=\"Float64\" Name=\"" << name << "\" format=\"ascii\">\n";
    //         for (int j = 0; j <= Ny; ++j)
    //             for (int i = 0; i <= Nx; ++i)
    //                 file << data[i + j * (Nx + 2)] << "\n";
    //         file << "</DataArray>\n";
    //     };

    //     write_array("Ux", host_ux);
    //     write_array("Uy", host_uy);
    //     write_array("C", host_c);
    //     write_array("P", host_p);
    //     write_array("Rho", host_rho);

    //     file << "</PointData>\n";
    //     file << "</Piece>\n</StructuredGrid>\n</VTKFile>\n";
    //     file.close();
    // }




    void VTSwriter::writeArray(std::ofstream &file, const char *name, const double *data,
                       int Nx, int Ny, int Nz = 0, bool is3D = false)
    {
        file << "<DataArray type=\"Float64\" Name=\"" << name << "\" format=\"ascii\">\n";

    

            for (CLIP_UINT i = m_domain->info.domainMinIdx[IDX_X]; i <= m_domain->info.domainMaxIdx[IDX_X]; i++)
                for (CLIP_UINT j = m_domain->info.domainMinIdx[IDX_Y]; j <= m_domain->info.domainMaxIdx[IDX_Y]; j++)
                    for (CLIP_UINT k = m_domain->info.domainMinIdx[IDX_Z]; k <= m_domain->info.domainMaxIdx[IDX_Z]; k++)
                        file << data[i + j * (Nx + 2) + k * (Nx + 2) * (Ny + 2)] << "\n";


        file << "</DataArray>\n";
    }

}
