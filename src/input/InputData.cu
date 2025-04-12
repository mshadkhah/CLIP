#include <InputData.cuh>


namespace clip {



    InputData::InputData(const std::string& filename)
    : m_filename(filename) {
        read_config();
    }



    void InputData::read_config(){
        std::cerr << "Reading Parameters:" << std::endl;
        read("D", D);
        read("tFinal", tFinal);
        read("finalStep", finalStep);
        read("noOutFiles", noOutFiles);
        read("Nx", Nx);
        read("Ny", Ny);
        read("Nz", Nz);
        read("X0", X0);
        read("Y0", Y0);
        read("Z0", Z0);
    
        read("Bo", Bo);
        read("Re", Re);
        read("We", We);
        read("Pe", Pe);
        read("Mo", Mo);

        read("rhoRatio", rhoRatio);
        read("muRatio", muRatio);

        read("gravity", gravity);
        read("interfaceWidth", interfaceWidth);
        read("mobility", mobility);
    }



    

    template <typename T>
    bool InputData::read_value(const std::string& varName, T& var) const {
        std::ifstream inputFile(m_filename);
        if (!inputFile.is_open()) {
            std::cerr << "Error opening config file: " << m_filename << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(inputFile, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);

                trim(key);
                trim(value);

                if (key == varName) {
                    std::istringstream iss(value);
                    iss >> var;
                    std::cout << varName << " = " << var << std::endl;
                    return true;
                }
            }
        }

        return false;
    }


    void InputData::trim(std::string& s) {
        size_t start = s.find_first_not_of(" \t");
        size_t end = s.find_last_not_of(" \t");
        if (start == std::string::npos) {
            s.clear();
        } else {
            s = s.substr(start, end - start + 1);
        }
    }


    bool InputData::read(const std::string& varName, CLIP_REAL& var) const {
        return read_value(varName, var);
    }
    
    bool InputData::read(const std::string& varName, CLIP_UINT& var) const {
        return read_value(varName, var);
    }
    
    bool InputData::read(const std::string& varName, bool& var) const {
        std::string valueStr;
        if (!read_value(varName, valueStr)) return false;
    
        if (valueStr == "true" || valueStr == "1") {
            var = true;
        } else if (valueStr == "false" || valueStr == "0") {
            var = false;
        } else {
            std::cerr << "Invalid boolean value for " << varName << std::endl;
            return false;
        }
    
        std::cout << varName << " = " << std::boolalpha << var << std::endl;
        return true;
    }

    
}



