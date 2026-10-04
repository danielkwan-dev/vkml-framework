# A project using vkml from outside, through add_subdirectory as the README
# shows: CI builds and runs it so that keeps working.
cmake_minimum_required(VERSION 3.25)
project(vkml_consumer LANGUAGES CXX)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../.. vkml)
add_executable(quickstart ${CMAKE_CURRENT_LIST_DIR}/../../examples/quickstart.cpp)
target_link_libraries(quickstart PRIVATE vkml::vkml)
