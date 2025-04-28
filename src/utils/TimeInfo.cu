#include <TimeInfo.cuh>

namespace clip
{

    TimeInfo::TimeInfo(InputData idata)
        : m_idata(idata) {}

    int TimeInfo::getCurrentStep() const
    {
        return m_info.currentStep;
    }

    double TimeInfo::getCurrentTime() const
    {
        return m_info.currentStep * m_info.dt;
    }

    void TimeInfo::setTimeStep(double dt)
    {
        m_info.dtIsSet = true;
        m_info.dt = dt;
    }

    void TimeInfo::increment()
    {
        m_info.currentStep++;
    }

    double TimeInfo::getFinalStep() const
    {
        return m_idata.params.finalStep;
    }

    TimeInfo::simInfo &clip::TimeInfo::getSimInfo()
    {
        return m_info;
    }

    const TimeInfo::simInfo &clip::TimeInfo::getSimInfo() const
    {
        return m_info;
    }

}
