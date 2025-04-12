#include <TimeInfo.cuh>


namespace clip {

    TimeInfo::TimeInfo(InputData idata)
    :m_idata(idata) {}

    
    int TimeInfo::getCurrentStep() const{
        return m_currentStep;
    }

    double TimeInfo::getCurrentTime() const{
        if (!m_dtIsSet)
            throw std::runtime_error("Time step not set before calling getCurrentTime()");
        return m_currentStep * m_dt;
    }

    void TimeInfo::setTimeStep(double dt){
        m_dtIsSet = true;
        m_dt = dt;
    }

    void TimeInfo::increment(){
        m_currentStep++;
    }

    double TimeInfo::getFinalStep() const{
        return m_idata.finalStep;
    }
}



