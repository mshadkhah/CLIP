#include <InputData.cuh>


namespace clip
{

    InputData::InputData(const std::string &filename)
        : m_filename(filename)
    {
        read_config();
    }

    void InputData::read_config()
    {
        std::cerr << "Reading Parameters:" << std::endl;
        // read("D", D);
        // read("tFinal", tFinal);
        // read("finalStep", finalStep);
        // read("noOutFiles", noOutFiles);
        // read("N", N);
        // read("Nx", Nx);
        // read("Ny", Ny);
        // read("Nz", Nz);
        // read("X0", X0);
        // read("Y0", Y0);
        // read("Z0", Z0);

        // read("Bo", Bo);
        // read("Re", Re);
        // read("We", We);
        // read("Pe", Pe);
        // read("Mo", Mo);

        // read("rhoRatio", rhoRatio);
        // read("muRatio", muRatio);

        // read("gravity", gravity);
        // read("interfaceWidth", interfaceWidth);
        // read("mobility", mobility);
    }

    InputData::CaseType InputData::caseTypeFromString(const std::string &str) {
        if (str == "drop") return CaseType::Drop;
        if (str == "bubble") return CaseType::Bubble;
        if (str == "jet") return CaseType::Jet;
        throw std::invalid_argument("Unknown case type: " + str);
    }
    

    template <typename T>
    bool InputData::read_array(const std::string &varName, std::vector<T> &arr) const
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

    bool InputData::read(const std::string &varName, std::vector<CLIP_UINT> &var) const
    {
        return read_array(varName, var);
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
    


    std::string InputData::getConfig(){
        return m_filename;
    }
    


    
}
