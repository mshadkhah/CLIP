#pragma once
#include <includes.h>
#include <InputData.cuh>


namespace clip {

    class TimeInfo {
        public:
            explicit TimeInfo(InputData idata);

            int getCurrentStep() const;
            double getCurrentTime() const;
            double getEndTime() const;
            double getFinalStep() const;
            void increment();
            void setTimeStep(double dt);
       


        
        private:
            double m_currentTime = 0;
            int m_currentStep = 0;
            double m_dt;
            bool m_dtIsSet = false;
            InputData m_idata;
        };


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



