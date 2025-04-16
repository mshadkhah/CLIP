#include <InputData.cuh>  
#include <TimeInfo.cuh>  
#include <NsAllen.cuh>
#include <Boundary.cuh>


int main() {

    clip::InputData input("../config.txt");
    clip::Boundary boundary(input);
    // clip::Equation eq(input);
    clip::TimeInfo ti(input);
    // clip::NSAllen eqn(input);

    
    while(ti.getCurrentStep() < ti.getFinalStep()){

        std::cout << "Testing ..." << std::endl;

        ti.increment();
    }


    





    

    return 0;
}
