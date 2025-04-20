#include <InputData.cuh>
#include <TimeInfo.cuh>
#include <NsAllen.cuh>
#include <Boundary.cuh>
#include <DataArray.cuh>

int main()
{

    clip::InputData input("../config.txt");

    clip::Domain domain(input);
    clip::DataArray DA(input, domain);

    DA.createVectors();

    clip::Boundary boundary(input, domain, DA);
    clip::NSAllen eqn(input, domain, DA, boundary);

    clip::TimeInfo ti(input);
    eqn.initializer();
    DA.updateDevice();

    // eqn.setVectors();
    // eqn.initializer();
    // eqn.collision();

    // while(ti.getCurrentStep() < ti.getFinalStep()){

    //     std::cout << "Testing ..." << std::endl;

    //     ti.increment();
    // }

    return 0;
}
