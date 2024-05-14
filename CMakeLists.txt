cmake_minimum_required(VERSION 2.8)

set(PROJ_NAME child_game_double)

project(${PROJ_NAME})
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

find_package(OpenCV REQUIRED)

add_executable(${PROJ_NAME}
main.cpp
match_pict.cpp
select_foregound.cpp)

target_include_directories(${PROJ_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJ_NAME} ${OpenCV_LIBS})
