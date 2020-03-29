# CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)

add_compile_options(-std=c++11)

# 项目信息
project (matrix_invert)

# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
# aux_source_directory(src DIR_SRCS)

# 添加外部库
find_package(benchmark REQUIRED)
find_package( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# 添加 invert 子目录
add_subdirectory(./src/invert)

# 设置环境变量
set(unittest ./src/invert_unittest.cpp)
set(benchmark ./src/invert_benchmark.cpp)

# 生成可执行目标
add_executable(invert_unittest ${unittest})
add_executable(invert_benchmark ${benchmark})

# 添加链接库
target_link_libraries(invert_unittest invert)
target_link_libraries(invert_unittest libgtest.a)
target_link_libraries(invert_unittest libpthread.so)

target_link_libraries(invert_benchmark invert)
target_link_libraries(invert_benchmark libpthread.so)
target_link_libraries(invert_benchmark benchmark::benchmark)
