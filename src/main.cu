#include <InputData.cuh>
#include "includes.h"
#include <TimeInfo.cuh>
#include <NsAllen.cuh>
#include <Boundary.cuh>
#include <DataArray.cuh>
#include "VTSwriter.cuh"
#include "Reporter.cuh"
#include "CheckPointer.cuh"
#include "Geometry.cuh"

int main(int argc, char *argv[])
{
    bool resumeFromCheckpoint = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "-resume")
        {
            resumeFromCheckpoint = true;
            clip::Logger::Info("Resume mode activated via command line.");
        }
    }

    // clip::InputData input("/home/mehdi/projects/CLIP/examples/2D/Bubble/config.txt");
    
    clip::InputData input("config.txt");

    clip::Domain domain(input);
    clip::Geometry geom(input);
    clip::Boundary boundary(input, domain);
    clip::DataArray DA(input, domain, boundary);
    DA.createVectors();

    

    // if (Geometry::sdf(geom, 0, x, y, z) <= 0)
    // printf("vel: %f \n",clip::Geometry::sdf(geom.getDeviceStruct(), 0, 32, 128, 32));

    clip::NSAllen eqn(input, domain, DA, boundary, geom);
    clip::TimeInfo ti(input);
    clip::VTSwriter output(DA, input, domain, ti, "test", "test");
    clip::Reporter report(DA, input, domain, ti);
    clip::CheckPointer chechpoint(DA, input, domain, ti, boundary);

    if (resumeFromCheckpoint)
    {
        chechpoint.load();
        eqn.macroscopic();
    }
    else
    {
        eqn.initialCondition();
        DA.updateDevice();
        eqn.deviceInitializer();
    }


    output.writeToFile();
  

    while (ti.getCurrentStep() < ti.getFinalStep())
    {
        eqn.solve();
        ti.increment();

        output.writeToFile();
        report.print();   
        chechpoint.save();
    }

    return 0;
}
