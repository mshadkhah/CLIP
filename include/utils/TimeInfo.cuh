#pragma once
#include <includes.h>
#include <InputData.cuh>

namespace clip {

class TimeInfo {
public:

    struct simInfo
    {
        double currentTime = 0;
        int currentStep = 0;
        double dt = 0;
        bool dtIsSet = false;
    };

    explicit TimeInfo(InputData idata);

    int getCurrentStep() const;
    double getCurrentTime() const;
    double getEndTime() const;
    double getFinalStep() const;
    void increment();
    void setTimeStep(double dt);

    simInfo& getSimInfo();
    const simInfo& getSimInfo() const;

private:
    InputData m_idata;
    simInfo m_info;
};

} // namespace clip
