#include <InputData.cuh>
#include <TimeInfo.cuh>
#include <NsAllen.cuh>
#include <Boundary.cuh>
#include <DataArray.cuh>
#include "VTSwriter.cuh"
#include "Reporter.cuh"
#include "CheckPointer.cuh"

int main()
{

    clip::InputData input("/home/mehdi/projects/CLIP/tests/2D/Bubble/config.txt");
    // clip::InputData input("config.txt");

    clip::Domain domain(input);
    clip::Boundary boundary(input, domain);
    clip::DataArray DA(input, domain, boundary);
    DA.createVectors();


    clip::NSAllen eqn(input, domain, DA, boundary);
    clip::TimeInfo ti(input);
    clip::VTSwriter output(DA, input, domain, ti, "test", "test");
    clip::Reporter report(DA, input, domain, ti);


    eqn.initialCondition();
    DA.updateDevice();
    eqn.deviceInitializer();


    clip::CheckPointer chechpoint(DA, input, domain, ti, boundary);


    while (ti.getCurrentStep() < ti.getFinalStep())
    {

        eqn.solve();

        output.writeToFile();
        ti.increment();
        report.print();
    }

    return 0;
}
