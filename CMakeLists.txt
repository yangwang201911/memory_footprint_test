cmake_minimum_required(VERSION 3.16)
project(mem_footprint)
set(SOURCE_FILES_MEM_FOOTPRINT
    main.cpp
)

find_package(OpenVINO REQUIRED)
add_executable(main ${SOURCE_FILES_MEM_FOOTPRINT})
target_link_libraries(main PRIVATE openvino::runtime)