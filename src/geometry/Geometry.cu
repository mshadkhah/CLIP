// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena


/**
 * @file Geometry.cu
 * @brief Reads, parses, and manages user-defined geometries for use in LBM simulations.
 * 
 * @details
 * This file implements the Geometry class, which reads geometry configurations from the input 
 * config file and prepares the geometry data (type, position, size) for both host and device use.
 * 
 * Supported geometry types include:
 * - Circle, Sphere
 * - Square, Cube
 * - Perturbation fields (e.g., for RTI)
 * 
 * Main responsibilities:
 * - Parse `[geometry]` blocks from the config file
 * - Populate `Entry` structs for each geometry
 * - Prepare a flat device-usable structure (`m_deviceGeometry`)
 * - Provide human-readable type names for debug/logging
 * 
 * Used by initialization routines to apply geometry-aware phase field conditions (e.g., SDF-based drop/bubble).
 * 
 * @author
 * Mehdi Shadkhah
 * 
 * @date
 * 2025
 */



#include "Geometry.cuh"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace clip
{


/// Constructs the Geometry manager using simulation input parameters.
/// @param idata Reference to input configuration

 
Geometry::Geometry(const InputData& idata)
    : m_idata(&idata)
{
    Logger::Info("Reading geometries...");
    readGeometries(geometries);
    fillDeviceGeometry();
    print();
    Logger::Success("Successfully read " + std::to_string(geometries.size()) + " geometries.");
}

/**
 * @brief Default destructor for Geometry.
 */
Geometry::~Geometry() {}


/**
 * @brief Reads geometry definitions from the input config file and populates the `geometries` vector.
 * 
 * @param geometries Reference to the vector where parsed geometry entries are stored.
 * @return true if at least one geometry was successfully read, false otherwise.
 */
bool Geometry::readGeometries(std::vector<Entry>& geometries)
{
    geometryObjects = 0;
    geometries.clear();
    
    std::ifstream inputFile(m_idata->getConfig());
    if (!inputFile.is_open())
    {
        Logger::Error("Error opening config file: " + m_idata->getConfig());
    }

    std::string line;
    bool inGeometryList = false;
    bool inBlock = false;
    Entry current;

    while (std::getline(inputFile, line))
    {
        trim(line);
        if (line.empty() || line[0] == '#')
            continue;

        // Start of geometry
        if (!inGeometryList && line.find("geometry") != std::string::npos && line.find('=') != std::string::npos)
        {
            inGeometryList = true;
            continue;
        }

        // ✅ End of geometry block
        if (inGeometryList && line == "]")
        {
            break;
        }

        if (inGeometryList)
        {
            if (line == "[")
                continue;
            if (line == "{")
            {
                inBlock = true;
                current = Entry{};
                continue;
            }
            if (line == "}" || line == "},")
            {
                inBlock = false;

                if (current.type == Type::Unknown)
                {
                    Logger::Error("Encountered a geometry with unknown type during parsing.");
                }

                geometries.push_back(current);
                geometryObjects++;
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

                    if (key == "type")
                        current.type = typeFromString(value);
                    else if (key == "center")
                    {
                        if (value.front() == '[') value.erase(0, 1);
                        if (value.back() == ']') value.pop_back();
                        std::stringstream ss(value);
                        std::string token;
                        int dim = 0;
                        while (std::getline(ss, token, ',') && dim < MAX_DIM)
                        {
                            trim(token);
                            current.center[dim++] = std::stod(token);
                        }
                    }
                    else if (key == "length")
                    {
                        if (value.front() == '[') value.erase(0, 1);
                        if (value.back() == ']') value.pop_back();
                        std::stringstream ss(value);
                        std::string token;
                        int dim = 0;
                        while (std::getline(ss, token, ',') && dim < MAX_DIM)
                        {
                            trim(token);
                            current.length[dim++] = std::stod(token);
                        }
                    }
                    else if (key == "radius")
                        current.radius = std::stod(value);
                    else if (key == "amplitude")
                        current.amplitude = std::stod(value);
                    else if (key == "id")
                        current.id = std::stoi(value);
                }
            }
        }
    }

    return !geometries.empty();
}


/**
 * @brief Fills the device-side geometry structure (`m_deviceGeometry`) with data parsed from the config file.
 */
void Geometry::fillDeviceGeometry()
{
    m_deviceGeometry.numGeometries = geometries.size();
    for (size_t i = 0; i < geometries.size(); ++i)
    {
        m_deviceGeometry.type[i] = static_cast<int>(geometries[i].type);
        for (int d = 0; d < MAX_DIM; ++d)
        {
            m_deviceGeometry.center[i][d] = geometries[i].center[d];
            m_deviceGeometry.length[i][d] = geometries[i].length[d];
        }
        m_deviceGeometry.radius[i] = geometries[i].radius;
        m_deviceGeometry.amplitude[i] = geometries[i].amplitude;
        m_deviceGeometry.id[i] = geometries[i].id;
    }
}

/**
 * @brief Converts a string to lowercase.
 * @param s Input string
 * @return Lowercase version of the string
 */
std::string Geometry::toLower(const std::string& s)
{
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}


/**
 * @brief Removes whitespace, tabs, and carriage returns from a string.
 * @param s String to be trimmed (modified in-place)
 */
void Geometry::trim(std::string& s)
{
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\r';
    }), s.end());
}


/**
 * @brief Converts a string into a geometry type (e.g., "circle" → Type::Circle).
 * @param str Geometry type as string
 * @return Corresponding Geometry::Type enum value
 */
Geometry::Type Geometry::typeFromString(const std::string& str)
{
    std::string lowerStr = toLower(str);

    if (lowerStr == "circle")
        return Type::Circle;
    if (lowerStr == "sphere")
        return Type::Sphere;
    if (lowerStr == "square")
        return Type::Square;
    if (lowerStr == "cube")
        return Type::Cube;
    if (lowerStr == "perturbation")
        return Type::Perturbation;

    return Type::Unknown;
}

/**
 * @brief Converts a Geometry::Type enum value into its string representation.
 * @param t Type enum value
 * @return Human-readable type name
 */
std::string Geometry::typeToString(Type t) const
{
    switch (t)
    {
    case Type::Circle: return "Circle";
    case Type::Sphere: return "Sphere";
    case Type::Square: return "Square";
    case Type::Cube: return "Cube";
    case Type::Perturbation: return "Perturbation";
    default: return "Unknown";
    }
}


/**
 * @brief Prints the parsed geometry list to standard output.
 */
void Geometry::print() const
{
    std::cout << "\nParsed Geometries:\n";
    for (size_t i = 0; i < geometryObjects; ++i)
    {
        const auto& geo = geometries[i];
        std::cout << "  Geometry " << i << ":\n";
        std::cout << "    Type: " << typeToString(geo.type) << "\n";
        std::cout << "    ID: " << geo.id << "\n";
        std::cout << "    Center: [" << geo.center[0] << ", " << geo.center[1] << ", " << geo.center[2] << "]\n";
        std::cout << "    Radius: " << geo.radius << "\n";
        std::cout << "    Length: [" << geo.length[0] << ", " << geo.length[1] << ", " << geo.length[2] << "]\n";
        std::cout << "    Amplitude: " << geo.amplitude << "\n\n";
    }
}



} // namespace clip
