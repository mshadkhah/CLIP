#include <InputData.cuh>
#include <TimeInfo.cuh>
#include <NsAllen.cuh>
#include <Boundary.cuh>
#include <DataArray.cuh>
#include "VTSwriter.cuh"
#include "Reporter.cuh"

int main()
{

    clip::InputData input("/home/mehdi/projects/CLIP/tests/2D/Bubble/config.txt");

    clip::Domain domain(input);
    clip::DataArray DA(input, domain);
    clip::Boundary boundary(input, domain, DA);

    DA.createVectors();


    clip::NSAllen eqn(input, domain, DA, boundary);
    clip::TimeInfo ti(input);
    clip::VTSwriter output(DA, input, domain, ti, "test", "test");
    clip::Reporter report(DA, input, domain, ti);


    eqn.initialCondition();
    DA.updateDevice();
    eqn.deviceInitializer();


    while (ti.getCurrentStep() < ti.getFinalStep())
    {

        eqn.solve();

        output.writeToFile();
        ti.increment();
        report.print();
    }

    return 0;
}
