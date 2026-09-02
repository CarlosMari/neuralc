cmake_minimum_required(VERSION 3.20)

project(neuralc
    VERSION 0.1.0
    LANGUAGES CXX
)

# ─────────────────────────────────────────────
# Project configuration
# ─────────────────────────────────────────────

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(
    ENABLE_SANITIZERS
    "Enable AddressSanitizer and UndefinedBehaviorSanitizer"
    OFF
)

# ─────────────────────────────────────────────
# Library
# ─────────────────────────────────────────────

add_library(neuralc
    src/tensor/storage.cpp
)

target_include_directories(neuralc
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(neuralc PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

# ─────────────────────────────────────────────
# Development executable
# ─────────────────────────────────────────────

add_executable(neuralc_demo
    src/main.cpp
)

target_link_libraries(neuralc_demo
    PRIVATE
        neuralc
)

# ─────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────

include(CTest)
enable_testing()

include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_executable(neuralc_tests
    tests/test_main.cpp
    tests/tensor/test_storage.cpp
)

target_link_libraries(neuralc_tests
    PRIVATE
        neuralc
        GTest::gtest_main
)

target_compile_options(neuralc_tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

include(GoogleTest)
gtest_discover_tests(neuralc_tests)

# ─────────────────────────────────────────────
# Sanitizers
# ─────────────────────────────────────────────

if(ENABLE_SANITIZERS)

    target_compile_options(neuralc PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )

    target_link_options(neuralc PRIVATE
        -fsanitize=address,undefined
    )

    target_compile_options(neuralc_demo PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )

    target_link_options(neuralc_demo PRIVATE
        -fsanitize=address,undefined
    )

    target_compile_options(neuralc_tests PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )

    target_link_options(neuralc_tests PRIVATE
        -fsanitize=address,undefined
    )

endif()

# ─────────────────────────────────────────────
# Linting
# ─────────────────────────────────────────────

find_program(CLANG_FORMAT clang-format)
find_program(CLANG_TIDY clang-tidy)

if(CLANG_FORMAT AND CLANG_TIDY)

    set(NEURALC_SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/src/tensor/storage.cpp
        ${PROJECT_SOURCE_DIR}/src/main.cpp
        ${PROJECT_SOURCE_DIR}/tests/test_main.cpp
        ${PROJECT_SOURCE_DIR}/tests/tensor/test_storage.cpp
    )

    add_custom_target(lint
        COMMAND ${CLANG_FORMAT}
            --dry-run
            --Werror
            ${NEURALC_SOURCE_FILES}

        COMMAND ${CLANG_TIDY}
            ${PROJECT_SOURCE_DIR}/src/tensor/storage.cpp
            --
            -I${PROJECT_SOURCE_DIR}/include
            -std=c++20

        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}

        COMMENT "Running clang-format and clang-tidy"
    )

else()

    message(WARNING
        "clang-format or clang-tidy not found. "
        "The lint target will not be available."
    )

endif()