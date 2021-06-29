cmake_minimum_required(VERSION 3.7)

project(wsn_all)

find_package(OpenMP REQUIRED)
find_package(MPI)

set(CMAKE_CXX_STANDARD 14)
include_directories(${PROJECT_SOURCE_DIR}/sim_wsn_all/cpp ${PROJECT_SOURCE_DIR}/sim_wsn_all/cpp/wsn ${PROJECT_SOURCE_DIR}/caravan-lib/json/single_include ${PROJECT_SOURCE_DIR}/caravan-lib/icecream)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
add_executable(wsn.out main.cpp)
target_link_libraries(wsn.out PRIVATE OpenMP::OpenMP_CXX)

add_executable(search.out search.cpp)
target_link_libraries(search.out PRIVATE OpenMP::OpenMP_CXX ${MPI_LIBRARIES})
target_include_directories(search.out PUBLIC ${MPI_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/caravan-lib)