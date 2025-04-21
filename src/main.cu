#include <InputData.cuh>
#include <TimeInfo.cuh>
#include <NsAllen.cuh>
#include <Boundary.cuh>
#include <DataArray.cuh>
#include "VTSwriter.cuh"

int main()
{

    clip::InputData input("../config.txt");

    clip::Domain domain(input);
    clip::DataArray DA(input, domain);

    DA.createVectors();

    clip::Boundary boundary(input, domain, DA);
    clip::NSAllen eqn(input, domain, DA, boundary);


    clip::TimeInfo ti(input);
    eqn.initialCondition();
    DA.updateDevice();
    eqn.deviceInitializer();


    clip::VTSwriter vi(DA, input, domain, ti, "test", "test");
    

    vi.WriteVTSBinaryFile();


    while(ti.getCurrentStep() < ti.getFinalStep()){

        eqn.solve();

        ti.increment();
    }

        DA.updateHost();
        vi.WriteVTSBinaryFile();




    return 0;
}
