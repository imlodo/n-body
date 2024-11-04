cmake_minimum_required(VERSION 3.10)  # Set the minimum version of CMake required. Adjust as needed.

project(NBodySimulation)  # Set the project name

# Set the C standard to C11
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

# Find the MPI package
find_package(MPI REQUIRED)

# Include directories for MPI
include_directories(${MPI_INCLUDE_PATH})

# Enable OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

# Add the executable
add_executable(nbody nbody.c)

# Link against MPI libraries
target_link_libraries(nbody ${MPI_C_LIBRARIES})

# If the MPI package added extra compilation flags
if(MPI_COMPILE_FLAGS)
    set_target_properties(nbody PROPERTIES
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

# If the MPI package added extra linking flags
if(MPI_LINK_FLAGS)
    set_target_properties(nbody PROPERTIES
        LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
