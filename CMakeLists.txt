cmake_minimum_required(VERSION 3.20)

project(neuralc
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_SANITIZERS "Enable sanitizers" OFF)

# Main Executable
add_executable(neuralc
    src/main.cpp
)

target_compile_options(neuralc PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

# Testing
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
)

target_link_libraries(neuralc_tests
    PRIVATE
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(neuralc_tests)

target_compile_options(neuralc_tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

# Sanitizers

option(ENABLE_SANITIZERS "Enable sanitizers" OFF)

if(ENABLE_SANITIZERS)
    target_compile_options(neuralc PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )

    target_link_options(neuralc PRIVATE
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
