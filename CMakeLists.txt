cmake_minimum_required(VERSION 3.10)
project(Seminar7Matrix)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

include_directories(include)
file(GLOB SOURCES "src/*.cpp")
add_executable(seminar7 ${SOURCES})
