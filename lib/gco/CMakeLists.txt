set(LIBRARY_NAME gco)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB LIBRARY_SRC_FILES src/*.cpp)

add_library(${LIBRARY_NAME} STATIC ${LIBRARY_SRC_FILES})
target_include_directories(${LIBRARY_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${LIBRARY_NAME}>
)
