#include <Boundary.cuh>
#include <includes.h>

namespace clip
{

    Boundary::Boundary(const InputData &idata, const Domain &domain, DataArray &DA)
        : m_idata(&idata), m_DA(&DA), m_domain(&domain)
    {

        dimBlock = m_DA->dimBlock;
        dimGrid = m_DA->dimGrid;

        readBoundaries(boundaries);
        updateFlags();
        print();

        m_DA->allocateOnDevice(dev_boundaryFlags, "dev_boundaryFlags", static_cast<CLIP_UINT>(Objects::MAX));

        // flagGenLauncher(dev_boundaryFlags, m_domain.info);
    }

    Boundary::~Boundary() {}

    Boundary::Objects Boundary::sideFromString(const std::string &str)
    {
        if (str == "x-")
            return Objects::XMinus;
        if (str == "x+")
            return Objects::XPlus;
        if (str == "y-")
            return Objects::YMinus;
        if (str == "y+")
            return Objects::YPlus;
        if (str == "z-")
            return Objects::ZMinus;
        if (str == "z+")
            return Objects::ZPlus;
        return Objects::Unknown;
    }

    Boundary::Type Boundary::typeFromString(const std::string &str)
    {
        if (str == "wall")
            return Type::Wall;
        if (str == "slip wall")
            return Type::SlipWall;
        if (str == "dirichlet")
            return Type::Dirichlet;
        if (str == "periodic")
            return Type::Periodic;
        return Type::Unknown;
    }

    std::string Boundary::toString(Objects side)
    {
        switch (side)
        {
        case Objects::XMinus:
            return "x-";
        case Objects::XPlus:
            return "x+";
        case Objects::YMinus:
            return "y-";
        case Objects::YPlus:
            return "y+";
        case Objects::ZMinus:
            return "z-";
        case Objects::ZPlus:
            return "z+";
        default:
            return "unknown";
        }
    }

    std::string Boundary::toString(Type type)
    {
        switch (type)
        {
        case Type::Wall:
            return "wall";
        case Type::SlipWall:
            return "slip wall";
        case Type::Dirichlet:
            return "dirichlet";
        case Type::Periodic:
            return "periodic";
        default:
            return "unknown";
        }
    }

    // Print utility
    void Boundary::print()
    {
        std::cout << "\nParsed Boundary Conditions:\n";
        for (size_t i = 0; i < boundaryObjects; ++i)
        {
            const auto &bc = boundaries[i];
            std::cout << "  Block " << i << ":\n";
            std::cout << "    Side: " << toString(bc.side) << "\n";
            std::cout << "    Type: " << toString(bc.BCtype) << "\n";
            std::cout << "    Value: " << bc.value << "\n";
            std::cout << "    : " << (bc.ifRefine ? "true" : "false") << "\n \n";
        }
    }

    bool Boundary::readBoundaries(std::vector<Entry> &boundaries)
    {
        boundaries.resize(20);
        boundaryObjects = 0;
        std::ifstream inputFile(m_idata->getConfig());
        if (!inputFile.is_open())
        {
            std::cerr << "Error opening config file: " << m_idata->getConfig() << std::endl;
            return false;
        }

        std::string line;
        bool inBoundaryList = false;
        bool inBlock = false;
        clip::Boundary::Entry current;

        while (std::getline(inputFile, line))
        {
            trim(line);
            if (line.empty() || line[0] == '#')
                continue;

            if (!inBoundaryList && line.find("boundary") != std::string::npos && line.find('=') != std::string::npos)
            {
                inBoundaryList = true;
                continue;
            }

            if (inBoundaryList)
            {
                if (line == "[")
                    continue;
                if (line == "{")
                {
                    inBlock = true;
                    current = clip::Boundary::Entry{};
                    continue;
                }
                if (line == "}" || line == "},")
                {
                    inBlock = false;
                    size_t index = static_cast<size_t>(current.side);
                    boundaries[index] = current;
                    boundaryObjects++;
                    continue;
                }

                if (inBlock)
                {
                    std::size_t pos = line.find('=');
                    if (pos != std::string::npos)
                    {
                        std::string key = line.substr(0, pos);
                        std::string value = line.substr(pos + 1);
                        trim(key);
                        trim(value);
                        value.erase(std::remove(value.begin(), value.end(), '"'), value.end());

                        if (key == "side")
                            current.side = clip::Boundary::sideFromString(value);
                        else if (key == "type")
                        {
                            current.BCtype = clip::Boundary::typeFromString(value);
                            size_t index = static_cast<size_t>(current.side);
                            Boundary::BCMap.types[index] = current.BCtype;
                        }
                        else if (key == "value")
                            current.value = std::stod(value);
                        else if (key == "ifRefine")
                            current.ifRefine = (value == "true" || value == "1");
                    }
                }
            }
        }

        // ✅ Validate that all required sides are defined
        std::set<clip::Boundary::Objects> expectedSides = {
            clip::Boundary::Objects::XMinus,
            clip::Boundary::Objects::XPlus,
            clip::Boundary::Objects::YMinus,
            clip::Boundary::Objects::YPlus};
        if (DIM == 3)
        {
            expectedSides.insert(clip::Boundary::Objects::ZMinus);
            expectedSides.insert(clip::Boundary::Objects::ZPlus);
        }

        std::set<clip::Boundary::Objects> foundSides;
        for (const auto &bc : boundaries)
        {
            foundSides.insert(bc.side);
        }

        std::vector<std::string> missing;
        for (const auto &side : expectedSides)
        {
            if (foundSides.find(side) == foundSides.end())
            {
                missing.push_back(clip::Boundary::toString(side));
            }
        }

        if (!missing.empty())
        {
            std::ostringstream oss;
            oss << "Missing boundary conditions for sides: ";
            for (size_t i = 0; i < missing.size(); ++i)
            {
                oss << missing[i];
                if (i != missing.size() - 1)
                    oss << ", ";
            }
            throw std::runtime_error(oss.str());
        }

        return !boundaries.empty();
    }

    void Boundary::trim(std::string &s)
    {
        size_t start = s.find_first_not_of(" \t");
        size_t end = s.find_last_not_of(" \t");
        if (start == std::string::npos)
        {
            s.clear();
        }
        else
        {
            s = s.substr(start, end - start + 1);
        }
    }

    __global__ void flagGen(CLIP_UINT *dev_flag, const Domain::DomainInfo domain)
    {
        const CLIP_UINT i = THREAD_IDX_X;
        const CLIP_UINT j = THREAD_IDX_Y;
        const CLIP_UINT k = (DIM == 3) ? THREAD_IDX_Z : 0;
        const CLIP_UINT idx_SCALAR = Domain::getIndex(domain, i, j, k);

        if (Domain::isInside<DIM>(domain, i, j, k))
        {
            // if(i == 0){
            //     dev_flag[idx_SCALAR] = boundaries[static_cast<CLIP_UINT>(Boundary::Objects::XMinus].BCtype);
            // }
            // else if(i == domain.extent[IDX_X]){
            //     dev_flag[idx_SCALAR] = static_cast<CLIP_UINT>(Boundary::Objects::XPlus);
            // }
            // else if(j == 0){
            //     dev_flag[idx_SCALAR] = static_cast<CLIP_UINT>(Boundary::Objects::YMinus);
            // }
            // else if(i == domain.extent[IDX_Y]){
            //     dev_flag[idx_SCALAR] = static_cast<CLIP_UINT>(Boundary::Objects::YPlus);
            // }

            // #ifdef ENABLE_2D
            // else if(k == 0){
            //     dev_flag[idx_SCALAR] = static_cast<CLIP_UINT>(Boundary::Objects::ZMinus);
            // }
            // else if(k == domain.extent[IDX_Z]){
            //     dev_flag[idx_SCALAR] = static_cast<CLIP_UINT>(Boundary::Objects::ZPlus);
            // }
            // #endif

            printf("index: i = %d\n", idx_SCALAR);
        }
    }

    void Boundary::flagGenLauncher(CLIP_UINT *dev_flag, const Domain::DomainInfo &domain)
    {
        flagGen<<<dimGrid, dimBlock>>>(dev_flag, domain);
        cudaDeviceSynchronize();
    }

    void Boundary::updateFlags()
    {

        for (size_t i = 0; i < boundaryObjects; ++i)
        {
            const auto &bc = boundaries[i];
            switch (bc.BCtype)
            {
            case Type::Wall:
                isWall = true;
                break;
            case Type::SlipWall:
                isSlipWall = true;
                break;
            case Type::Dirichlet:
                isSlipWall = true;
                break;
            case Type::Periodic:
                isPeriodic = true;
                break;
            }
        }
    }
}