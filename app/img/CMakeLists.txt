set(BINARY_NAME img-gco)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB BINARY_SRC_FILES src/*.cpp)

add_executable(${BINARY_NAME} ${BINARY_SRC_FILES})

# Greet project's libraries (NOTE, the order is important)
set(BINARY_LINK_LIBS
    # Project libs
    gco
    
    )

target_link_libraries(${BINARY_NAME} ${BINARY_LINK_LIBS} ${OpenCV_LIBS} ${Boost_LIBRARIES} pthread X11)
