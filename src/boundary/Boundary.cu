// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file
 * @brief Implements the Boundary class methods responsible for reading, parsing, and managing
 *        boundary conditions in the CLIP LBM framework.
 *
 * Key functionalities:
 * - Parses structured boundary blocks from the config file
 * - Maps user-defined types and sides to enums
 * - Fills boundary condition maps used in solver kernels
 * - Prints parsed boundary configuration
 * - Sets internal flags for BC types for fast checking
 */

#include <Boundary.cuh>
#include <includes.h>

namespace clip
{

    // Constructor: Initializes Boundary from input and domain, then reads and prints boundaries.
    Boundary::Boundary(const InputData &idata, const Domain &domain)
        : m_idata(&idata), m_domain(&domain)
    {

        readBoundaries(boundaries);
        updateFlags();
        print();
    }

    // Destructor
    Boundary::~Boundary() {}

    // Converts string to Boundary::Objects enum
    Boundary::Objects Boundary::sideFromString(const std::string &str)
    {
        std::string lowerStr = toLower(str); // << make it lowercase
        if (lowerStr == "x-")
            return Objects::XMinus;
        if (lowerStr == "x+")
            return Objects::XPlus;
        if (lowerStr == "y-")
            return Objects::YMinus;
        if (lowerStr == "y+")
            return Objects::YPlus;
        if (lowerStr == "z-")
            return Objects::ZMinus;
        if (lowerStr == "z+")
            return Objects::ZPlus;
        return Objects::Unknown;
    }

    // Converts string to Boundary::Type enum
    Boundary::Type Boundary::typeFromString(const std::string &str)
    {
        std::string lowerStr = toLower(str); // << make it lowercase
        if (lowerStr == "wall")
            return Type::Wall;
        if (lowerStr == "slip wall")
            return Type::SlipWall;
        if (lowerStr == "free convect")
            return Type::FreeConvect;
        if (lowerStr == "periodic")
            return Type::Periodic;
        if (lowerStr == "neumann")
            return Type::Neumann;
        if (lowerStr == "velocity")
            return Type::Velocity;
        if (lowerStr == "do nothing")
            return Type::DoNothing;
        return Type::Unknown;
    }

    // Converts enum side to string
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

    // Converts enum type to string
    std::string Boundary::toString(Type type)
    {
        switch (type)
        {
        case Type::Wall:
            return "wall";
        case Type::SlipWall:
            return "slip wall";
        case Type::FreeConvect:
            return "free convect";
        case Type::Periodic:
            return "periodic";
        case Type::Neumann:
            return "neumann";
        case Type::Velocity:
            return "velocity";
        case Type::DoNothing:
            return "do nothing";
        default:
            return "unknown";
        }
    }

    // Prints all parsed boundary entries to console
    void Boundary::print()
    {
        std::cout << "\nParsed Boundary Conditions:\n";
        for (size_t i = 0; i < boundaryObjects; ++i)
        {
            const auto &bc = boundaries[i];
            std::cout << "  Block " << i << ":\n";
            std::cout << "    Side: " << toString(bc.side) << "\n";
            std::cout << "    Type: " << toString(bc.BCtype) << "\n";
            std::cout << "    Value: [";
            for (int d = 0; d < MAX_DIM; ++d)
            {
                std::cout << bc.value[d];
                if (d != MAX_DIM - 1)
                    std::cout << ", ";
            }
            std::cout << "]\n";
            std::cout << "    Refinement: " << (bc.ifRefine ? "true" : "false") << "\n\n";
        }
    }

    // Reads boundary blocks from the config file and populates BCMap
    bool Boundary::readBoundaries(std::vector<Entry> &boundaries)
    {
        boundaries.resize(20);
        boundaryObjects = 0;
        std::ifstream inputFile(m_idata->getConfig());
        if (!inputFile.is_open())
        {
            Logger::Error("Error opening config file: " + m_idata->getConfig());
            return false;
        }

        std::string line;
        bool inBoundaryList = false;
        bool inBlock = false;
        clip::Boundary::Entry current;

        while (std::getline(inputFile, line))
        {
            trim(line); // << ✅ unified cleanup

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
                        {
                            if (value.front() == '[')
                                value.erase(0, 1);
                            if (value.back() == ']')
                                value.pop_back();
                            std::stringstream ss(value);
                            std::string token;
                            int dim = 0;
                            while (std::getline(ss, token, ',') && dim < MAX_DIM)
                            {
                                trim(token);
                                current.value[dim] = std::stod(token);
                                size_t index = static_cast<size_t>(current.side);
                                Boundary::BCMap.val[index][dim] = std::stod(token);
                                dim++;
                            }
                        }
                        else if (key == "ifRefine")
                            current.ifRefine = (value == "true" || value == "1");
                    }
                }
            }
        }

        // Check if all sides are defined
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

    // Removes whitespace characters from a string (used in config parsing)
    void Boundary::trim(std::string &s)
    {
        // Remove spaces, tabs, and carriage returns everywhere
        s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c)
                               { return c == ' ' || c == '\t' || c == '\r'; }),
                s.end());
    }

    // Updates boundary flags (e.g., isWall, isPeriodic) after parsing
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
            case Type::FreeConvect:
                isFreeConvect = true;
                break;
            case Type::Neumann:
                isNeumann = true;
                break;
            case Type::Periodic:
                isPeriodic = true;
                break;
            case Type::Velocity:
                isVelocity = true;
                break;
            }
        }
    }

    // Converts a string to lowercase
    std::string Boundary::toLower(const std::string &s)
    {
        std::string result = s;
        std::transform(result.begin(), result.end(), result.begin(),
                       [](unsigned char c)
                       { return std::tolower(c); });
        return result;
    }

}