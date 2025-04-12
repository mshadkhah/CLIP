#include <InputData.cuh>  
#include <TimeInfo.cuh>  
#include <NsAllen.cuh>


int main() {

    clip::InputData input("../config.txt");
    clip::TimeInfo ti(input);
    clip::NSAllen eqn(input);

    
    while(ti.getCurrentStep() < ti.getFinalStep()){

        std::cout << "Testing ..." << std::endl;

        ti.increment();
    }


    





    

    return 0;
}
