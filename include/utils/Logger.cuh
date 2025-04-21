#pragma once
#include <iostream>
#include <string>
#include <stdexcept>

namespace clip {

class Logger {
public:
    // Info message
    static void Info(const std::string& message) {
        std::cout << "\033[1;34m[Info] \033[0m" << message << std::endl;
    }

    // Warning message
    static void Warning(const std::string& message) {
        std::cout << "\033[1;33m[Warning] \033[0m" << message << std::endl;
    }

    // Error message with optional throw
    static void Error(const std::string& message, bool shouldThrow = true) {
        std::cerr << "\033[1;31m[Error] \033[0m" << message << std::endl;
        if (shouldThrow) {
            throw std::runtime_error(message);
        }
    }

    // Success message
    static void Success(const std::string& message) {
        std::cout << "\033[1;32m[Success] \033[0m" << message << std::endl;
    }

    // Debug (optional toggle via preprocessor)
#ifdef CLIP_DEBUG
    static void Debug(const std::string& message) {
        std::cout << "\033[1;35m[Debug] \033[0m" << message << std::endl;
    }
#endif
};

} 
