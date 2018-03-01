cmake_minimum_required(VERSION 2.8)
set(PROJECT_NAME_STR gco-dp)
project(${PROJECT_NAME_STR} C CXX)

find_package(OpenCV REQUIRED)

set (Boost_NO_SYSTEM_PATHS ON)
set (Boost_USE_MULTITHREADED ON)
set (Boost_USE_STATIC_LIBS ON)
set (Boost_USE_STATIC_RUNTIME OFF)
set (BOOST_ALL_DYN_LINK OFF)

find_package (Boost 1.55.0 REQUIRED COMPONENTS log filesystem)
if (Boost_FOUND)
  include_directories (SYSTEM ${Boost_INCLUDE_DIR})
endif ()


if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_definitions(-std=gnu++11 -Wall -Weffc++ -Wno-deprecated)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(MSVC_COMPILER_DEFS "-D_CRT_SECURE_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS -D_VARIADIC_MAX=10")
    add_definitions(${MSVC_COMPILER_DEFS})
    add_definitions(/MP)
endif()

set(APP_PROJECTS_DIR ${PROJECT_SOURCE_DIR}/app)
set(LIB_PROJECTS_DIR ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(${LIB_PROJECTS_DIR}/gco)
add_subdirectory(${APP_PROJECTS_DIR}/example)
add_subdirectory(${APP_PROJECTS_DIR}/img)

