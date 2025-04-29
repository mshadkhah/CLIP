#include <InputData.cuh>

namespace clip
{

    InputData::InputData(const std::string &filename)
        : m_filename(filename)
    {
        Logger::Info("Reading input parameters...");
        read_config();
        Logger::Success("Input parameters loaded successfully.");
    }

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

    bool InputData::read(const std::string &varName, CLIP_REAL &var) const
    {
        return read_value(varName, var);
    }

    bool InputData::read(const std::string &varName, CLIP_UINT &var) const
    {
        return read_value(varName, var);
    }

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

    template <typename T, std::size_t N>
    bool InputData::read(const std::string &varName, T (&arr)[N]) const
    {
        return read_array(varName, arr);
    }

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

    std::string InputData::getConfig() const
    {
        return m_filename;
    }

}
