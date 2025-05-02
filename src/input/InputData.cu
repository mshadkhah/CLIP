// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file InputData.cu
 * @brief Parses and stores simulation configuration parameters from a user-defined config file.
 *
 * @details
 * This file defines the `InputData` class, which reads simulation parameters such as
 * Reynolds number, Weber number, grid size, time control, and boundary condition settings.
 * It handles type conversion, default handling, and special logic depending on the simulation case.
 *
 * Supported parameter types:
 * - Scalars (e.g., `tFinal`, `Re`, `We`, `referenceLength`)
 * - Vectors (e.g., `gravity`, `N`)
 * - Arrays (e.g., `[tauL, tauH]`)
 * - Enums (e.g., `caseType` as Drop, Bubble, Jet, RTI)
 *
 * ## Key Responsibilities
 * - Parse key-value pairs from config
 * - Apply case-specific physics initialization (e.g., compute `muH`, `sigma`, `tau`)
 * - Convert strings to enums and typed arrays
 * - Provide access to parsed data via the `params` struct
 *
 * Used throughout the solver to configure LBM model parameters, geometry scaling, time stepping, and more.
 *
 * @author
 * Mehdi Shadkhah
 *
 * @date
 * 2025
 */

#include <InputData.cuh>

namespace clip
{

/// Constructs an InputData object and loads simulation parameters from a config file.
/// filename Path to the configuration file

    InputData::InputData(const std::string &filename)
        : m_filename(filename)
    {
        Logger::Info("Reading input parameters...");
        read_config();
        Logger::Success("Input parameters loaded successfully.");
    }

    /**
     * @brief Reads all simulation parameters from the config file and fills the `params` structure.
     *
     * Initializes derived quantities based on the selected case type (e.g., `Drop`, `Bubble`, `RTI`, `Jet`),
     * including `muH`, `muL`, `sigma`, `tauH`, `tauL`, and phase-field constants.
     */
    void InputData::read_config()
    {

        read("case", params.caseType);
        read("tFinal", params.tFinal);
        read("finalStep", params.finalStep);
        read("outputInterval", params.outputInterval);
        read("checkpointInterval", params.checkpointInterval);
        read("checkpointCopy", params.checkpointCopy);
        read("reportInterval", params.reportInterval);
        read("N", params.N);
        read("referenceLength", params.referenceLength);
        read("interfaceWidth", params.interfaceWidth);
        read("gravity", params.gravity);
        read("mobility", params.mobility);
        read("muRatio", params.muRatio);
        read("rhoRatio", params.rhoRatio);
        read("amplitude", params.amplitude);

        params.RhoH = 1.0;
        params.RhoL = params.RhoH / params.rhoRatio;

        if (params.caseType == CaseType::Bubble || params.caseType == CaseType::Drop)
        {
            read("We", params.We);
            read("Re", params.Re);

            // params.sigma = (params.gravity * (params.RhoH - params.RhoL) * params.D * params.D) / params.Bo;
            params.sigma = (params.RhoH * params.gravity[IDX_Y] * params.referenceLength * params.referenceLength) / params.We;
            params.muH = sqrt(params.gravity[IDX_Y] * params.RhoH * (params.RhoH - params.RhoL) * params.referenceLength * params.referenceLength * params.referenceLength) / params.Re;
        }

        else if (params.caseType == CaseType::RTI)
        {
            read("Ca", params.Ca);
            read("Pe", params.Pe);
            read("Re", params.Re);

            params.muH = (params.RhoH * sqrt(params.gravity[IDX_Y] * params.referenceLength) * params.referenceLength) / params.Re;
            params.sigma = (params.muH * sqrt(params.gravity[IDX_Y] * params.referenceLength)) / params.Ca;
            params.mobility = (sqrt(params.gravity[IDX_Y] * params.referenceLength) * params.referenceLength) / params.Pe;
        }
        else if (params.caseType == CaseType::Jet)
        {

            read("We", params.We);
            read("Re", params.Re);
            read("referenceVelocity", params.referenceVelocity);

            params.sigma = (params.RhoH * params.referenceVelocity * params.referenceVelocity * params.referenceLength) / params.We;
            params.muH = (params.RhoH * params.referenceVelocity * params.referenceLength) / params.Re;
        }

        params.muL = params.muH / params.muRatio;
        params.tauH = 3.0 * (params.muH / params.RhoH);
        params.tauL = 3.0 * (params.muL / params.RhoL);
        params.kConstant = 1.50 * params.sigma * params.interfaceWidth;
        params.betaConstant = 8.0 * params.sigma / params.interfaceWidth;
    }

    /**
     * @brief Converts a string to the corresponding CaseType enum.
     * @param str Case name as lowercase string
     * @return Corresponding CaseType enum value
     * @throws std::invalid_argument if the case name is unknown
     */
    InputData::CaseType InputData::caseTypeFromString(const std::string &str)
    {
        if (str == "drop")
            return CaseType::Drop;
        if (str == "bubble")
            return CaseType::Bubble;
        if (str == "jet")
            return CaseType::Jet;
        if (str == "rti")
            return CaseType::RTI;
        throw std::invalid_argument("Unknown case type: " + str);
    }

    /**
     * @brief Reads an array of N values from the config file and stores them in a fixed-size array.
     * @tparam T Type of elements (CLIP_REAL or CLIP_UINT)
     * @tparam N Size of the array
     * @param varName Name of the variable to read
     * @param arr Output array
     * @return true if the value was successfully read
     */
    template <typename T, std::size_t N>
    bool InputData::read_array(const std::string &varName, T (&arr)[N]) const
    {
        std::ifstream inputFile(m_filename);
        if (!inputFile.is_open())
        {
            std::cerr << "Error opening config file: " << m_filename << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(inputFile, line))
        {
            if (line.empty() || line[0] == '#')
                continue;

            std::size_t pos = line.find('=');
            if (pos != std::string::npos)
            {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                trim(key);
                trim(value);

                if (key == varName)
                {
                    std::fill(std::begin(arr), std::end(arr), static_cast<T>(0));

                    value.erase(std::remove(value.begin(), value.end(), '['), value.end());
                    value.erase(std::remove(value.begin(), value.end(), ']'), value.end());

                    std::stringstream ss(value);
                    std::string token;
                    std::size_t count = 0;

                    while (std::getline(ss, token, ',') && count < N)
                    {
                        trim(token);
                        if constexpr (std::is_same<T, CLIP_UINT>::value)
                            arr[count++] = static_cast<T>(std::stoul(token));
                        else if constexpr (std::is_same<T, CLIP_REAL>::value)
                            arr[count++] = static_cast<T>(std::stod(token));
                    }

                    // Print the result
                    std::cout << varName << " = [";
                    for (std::size_t i = 0; i < N; ++i)
                    {
                        std::cout << arr[i];
                        if (i < N - 1)
                            std::cout << ", ";
                    }
                    std::cout << "]\n";

                    if (count < N)
                    {
                        std::cerr << "Warning: Only " << count << " values provided for " << varName
                                  << "; remaining " << (N - count) << " set to 0.\n";
                    }

                    return true;
                }
            }
        }

        return false;
    }

    /**
     * @brief Reads a vector of values from the config file.
     * @tparam T Type of elements (CLIP_REAL or CLIP_UINT)
     * @param varName Name of the variable to read
     * @param arr Output vector
     * @return true if the value was successfully read
     */
    template <typename T>
    bool InputData::read_vector(const std::string &varName, std::vector<T> &arr) const
    {
        std::ifstream inputFile(m_filename);
        if (!inputFile.is_open())
        {
            std::cerr << "Error opening config file: " << m_filename << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(inputFile, line))
        {
            if (line.empty() || line[0] == '#')
                continue;

            std::size_t pos = line.find('=');
            if (pos != std::string::npos)
            {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);

                trim(key);
                trim(value);

                if (key == varName)
                {
                    arr.clear();
                    // Remove square brackets
                    value.erase(std::remove(value.begin(), value.end(), '['), value.end());
                    value.erase(std::remove(value.begin(), value.end(), ']'), value.end());

                    std::stringstream ss(value);
                    std::string token;
                    while (std::getline(ss, token, ','))
                    {
                        trim(token);
                        if constexpr (std::is_same<T, CLIP_UINT>::value)
                            arr.push_back(static_cast<T>(std::stoul(token)));
                        else if constexpr (std::is_same<T, CLIP_REAL>::value)
                            arr.push_back(static_cast<T>(std::stod(token)));
                        else
                            static_assert(sizeof(T) == 0, "Unsupported type for read_array");
                    }

                    std::cout << varName << " = [";
                    for (size_t i = 0; i < arr.size(); ++i)
                        std::cout << arr[i] << (i < arr.size() - 1 ? ", " : "");
                    std::cout << "]" << std::endl;

                    return true;
                }
            }
        }

        return false;
    }

    /**
     * @brief Reads a scalar value from the config file.
     * @tparam T Type of the variable (CLIP_REAL, CLIP_UINT, etc.)
     * @param varName Name of the variable to read
     * @param var Output value
     * @return true if the value was successfully read
     */
    template <typename T>
    bool InputData::read_value(const std::string &varName, T &var) const
    {
        std::ifstream inputFile(m_filename);
        if (!inputFile.is_open())
        {
            std::cerr << "Error opening config file: " << m_filename << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(inputFile, line))
        {
            if (line.empty() || line[0] == '#')
                continue;

            std::size_t pos = line.find('=');
            if (pos != std::string::npos)
            {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);

                trim(key);
                trim(value);

                if (key == varName)
                {
                    std::istringstream iss(value);
                    iss >> var;
                    std::cout << varName << " = " << var << std::endl;
                    return true;
                }
            }
        }

        return false;
    }

    /**
     * @brief Trims whitespace from the beginning and end of a string.
     * @param s String to be trimmed (modified in-place)
     */
    void InputData::trim(std::string &s)
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

    /**
     * @brief Reads a CLIP_REAL variable from the config file.
     * @param varName Name of the variable
     * @param var Output value
     * @return true if successful
     */
    bool InputData::read(const std::string &varName, CLIP_REAL &var) const
    {
        return read_value(varName, var);
    }

    /**
     * @brief Reads a CLIP_UINT variable from the config file.
     * @param varName Name of the variable
     * @param var Output value
     * @return true if successful
     */
    bool InputData::read(const std::string &varName, CLIP_UINT &var) const
    {
        return read_value(varName, var);
    }

    /**
     * @brief Reads a boolean variable from the config file (expects "true"/"false" or "1"/"0").
     * @param varName Name of the variable
     * @param var Output value
     * @return true if successful
     */
    bool InputData::read(const std::string &varName, bool &var) const
    {
        std::string valueStr;
        if (!read_value(varName, valueStr))
            return false;

        if (valueStr == "true" || valueStr == "1")
        {
            var = true;
        }
        else if (valueStr == "false" || valueStr == "0")
        {
            var = false;
        }
        else
        {
            std::cerr << "Invalid boolean value for " << varName << std::endl;
            return false;
        }

        std::cout << varName << " = " << std::boolalpha << var << std::endl;
        return true;
    }

    /**
     * @brief Reads a fixed-size array of values from the config file.
     * @tparam T Type of array elements
     * @tparam N Array size
     * @param varName Name of the variable
     * @param arr Output array
     * @return true if successful
     */
    template <typename T, std::size_t N>
    bool InputData::read(const std::string &varName, T (&arr)[N]) const
    {
        return read_array(varName, arr);
    }

    /**
     * @brief Reads the simulation case type as a string and converts it to a CaseType enum.
     * @param varName Should be "case"
     * @param caseType Output CaseType value
     * @return true if successful
     */
    bool InputData::read(const std::string &varName, CaseType &caseType) const
    {
        std::string str;
        if (read_value(varName, str))
        {
            std::transform(str.begin(), str.end(), str.begin(), ::tolower);
            caseType = caseTypeFromString(str);
            return true;
        }
        else
        {
            std::cerr << "Warning: caseType not specified in config. Using default." << std::endl;
            return false;
        }
    }

    /**
     * @brief Returns the path to the currently loaded config file.
     * @return File path as a string
     */
    std::string InputData::getConfig() const
    {
        return m_filename;
    }

}
