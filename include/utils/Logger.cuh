// Copyright (c) 2020–2025 Mehdi Shadkhah
// SPDX-License-Identifier: BSD-3-Clause
// Part of CLIP: A CUDA-Accelerated LBM Framework for Interfacial Phenomena

/**
 * @file
 * @brief Defines the Logger class for standardized colored console output and error handling in the CLIP framework.
 *
 * The Logger provides:
 * - Informational, warning, success, and error messages
 * - ANSI escape codes for colored terminal output
 * - Optional runtime exceptions for errors
 * - Debug logging with compile-time toggle (`CLIP_DEBUG`)
 *
 * This utility improves terminal readability and helps track runtime diagnostics in CLIP simulations.
 */

#pragma once
#include <iostream>
#include <string>
#include <stdexcept>

namespace clip
{

    /**
     * @brief Utility class for formatted console output with color and severity labels.
     *
     * The Logger supports standard output levels:
     * - Info: General status messages
     * - Warning: Non-critical issues
     * - Error: Fatal errors with optional exception
     * - Success: Highlighted confirmation
     * - Debug: Conditionally compiled developer-level diagnostics
     */
    class Logger
    {
    public:
        /**
         * @brief Prints an informational message to stdout.
         * @param message The message to display
         */
        static void Info(const std::string &message)
        {
            std::cout << "\033[1;34m[Info] \033[0m" << message << std::endl;
        }
        /**
         * @brief Prints a warning message to stdout in yellow.
         * @param message The message to display
         */
        static void Warning(const std::string &message)
        {
            std::cout << "\033[1;33m[Warning] \033[0m" << message << std::endl;
        }

        /**
         * @brief Prints an error message to stderr and optionally throws an exception.
         * @param message The error message
         * @param shouldThrow Whether to throw std::runtime_error (default = true)
         */
        static void Error(const std::string &message, bool shouldThrow = true)
        {
            std::cerr << "\033[1;31m[Error] \033[0m" << message << std::endl;
            if (shouldThrow)
            {
                throw std::runtime_error(message);
            }
        }

        /**
         * @brief Prints a success message to stdout in green.
         * @param message The message to display
         */
        static void Success(const std::string &message)
        {
            std::cout << "\033[1;32m[Success] \033[0m" << message << std::endl;
        }

        // Debug (optional toggle via preprocessor)
#ifdef CLIP_DEBUG
        static void Debug(const std::string &message)
        {
            std::cout << "\033[1;35m[Debug] \033[0m" << message << std::endl;
        }
#endif
    };

}
