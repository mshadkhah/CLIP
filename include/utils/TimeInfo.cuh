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
    
}



