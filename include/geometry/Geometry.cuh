#pragma once

#include "InputData.cuh"
#include "includes.h"
#include "Domain.cuh"
#include <string>

#define MAX_GEOMETRIES 16

namespace clip
{

class Geometry
{
public:
    explicit Geometry(const InputData& idata);

    ~Geometry();

    enum class Type
    {
        Circle = 0,
        Sphere = 1,
        Square = 2,
        Cube = 3,
        Perturbation = 4,
        Unknown = 5,
        MAX = 6
    };

    struct Entry
    {
        Type type = Type::Unknown;
        CLIP_REAL center[MAX_DIM] = {0.0, 0.0, 0.0};
        CLIP_REAL length[MAX_DIM] = {0.0, 0.0, 0.0};
        CLIP_REAL radius = 0.0;
        CLIP_REAL amplitude = 0.0;
        int id = -1;
    };

    struct GeometryDevice
    {
        CLIP_INT numGeometries = 0;
        CLIP_INT type[MAX_GEOMETRIES];
        CLIP_REAL center[MAX_GEOMETRIES][MAX_DIM];
        CLIP_REAL length[MAX_GEOMETRIES][MAX_DIM];
        CLIP_REAL radius[MAX_GEOMETRIES];
        CLIP_REAL amplitude[MAX_GEOMETRIES];
        CLIP_INT id[MAX_GEOMETRIES];
    };

    __device__ __host__ inline
    CLIP_REAL static sdf(const GeometryDevice& geo, CLIP_INT id, CLIP_REAL x, CLIP_REAL y, CLIP_REAL z)
    {
        for (int i = 0; i < geo.numGeometries; ++i)
        {
            if (geo.id[i] == id)
            {
                switch (geo.type[i])
                {
                case static_cast<CLIP_INT>(Type::Circle):
                {
                    CLIP_REAL dx = x - geo.center[i][0];
                    CLIP_REAL dy = y - geo.center[i][1];
                    return sqrt(dx * dx + dy * dy) - geo.radius[i];
                }
                case static_cast<CLIP_INT>(Type::Sphere):
                {
                    CLIP_REAL dx = x - geo.center[i][0];
                    CLIP_REAL dy = y - geo.center[i][1];
                    CLIP_REAL dz = z - geo.center[i][2];
                    return sqrt(dx * dx + dy * dy + dz * dz) - geo.radius[i];
                }
                case static_cast<CLIP_INT>(Type::Square):
                {
                    CLIP_REAL dx = fabs(x - geo.center[i][0]) - geo.length[i][0] * 0.5;
                    CLIP_REAL dy = fabs(y - geo.center[i][1]) - geo.length[i][1] * 0.5;
                    CLIP_REAL ax = max(dx, 0.0);
                    CLIP_REAL ay = max(dy, 0.0);
                    CLIP_REAL outside = sqrt(ax * ax + ay * ay);
                    CLIP_REAL inside = min(max(dx, dy), 0.0);
                    return outside + inside;
                }
                case static_cast<CLIP_INT>(Type::Cube):
                {
                    CLIP_REAL dx = fabs(x - geo.center[i][0]) - geo.length[i][0] * 0.5;
                    CLIP_REAL dy = fabs(y - geo.center[i][1]) - geo.length[i][1] * 0.5;
                    CLIP_REAL dz = fabs(z - geo.center[i][2]) - geo.length[i][2] * 0.5;
                    CLIP_REAL ax = max(dx, 0.0);
                    CLIP_REAL ay = max(dy, 0.0);
                    CLIP_REAL az = max(dz, 0.0);
                    CLIP_REAL outside = sqrt(ax * ax + ay * ay + az * az);
                    CLIP_REAL inside = min(max(max(dx, dy), dz), 0.0);
                    return outside + inside;
                }
                case static_cast<CLIP_INT>(Type::Perturbation):
                {
    #ifdef ENABLE_2D
                    const CLIP_REAL perturbation = geo.amplitude[i] * geo.center[i][0] * cos(2.0 * M_PI * x / geo.center[i][0]);
    #elif defined(ENABLE_3D)
                    const CLIP_REAL perturbation = geo.amplitude[i] * geo.center[i][0] * (cos(2.0 * M_PI * x / geo.center[i][0]) + cos(2.0 * M_PI * z / geo.center[i][2]));
    #endif
                    const CLIP_REAL yShift = geo.center[i][1] - perturbation;
                    return y - yShift;
                }
                }
            }
        }
    
        return 1e10; // if no matching object
    }

    const GeometryDevice& getDeviceStruct() const { return m_deviceGeometry; }

    // NEW
    void print() const;

private:
    const InputData* m_idata;
    std::vector<Entry> geometries;
    GeometryDevice m_deviceGeometry;
    CLIP_UINT geometryObjects = 0;

    bool readGeometries(std::vector<Entry>& geometries);
    void fillDeviceGeometry();
    std::string toLower(const std::string& s);
    void trim(std::string& s);
    Type typeFromString(const std::string& str);
    std::string typeToString(Type t) const;
};

} // namespace clip
