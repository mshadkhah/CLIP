// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena


#pragma once
#include "includes.h"
#include "InputData.cuh"
#include "DataTypes.cuh"
#include "DataArray.cuh"
#include "Domain.cuh"
#include "TimeInfo.cuh"

namespace clip
{

    class Reporter
    {
    public:
        explicit Reporter(DataArray &DA, const InputData &idata, const Domain &domain, const TimeInfo &ti);

        void print();


    private:
        const Domain *m_domain;
        const InputData *m_idata;
        const TimeInfo *m_ti;
        DataArray *m_DA;

        CLIP_REAL sum(CLIP_REAL* dev_data, CLIP_UINT dof);

    };

}
