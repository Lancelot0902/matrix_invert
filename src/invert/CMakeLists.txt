# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_LIB_SRCS 变量
aux_source_directory(. DIR_LIB_SRCS)

find_package(OpenCV REQUIRED)

# 生成链接库
add_library (invert ${DIR_LIB_SRCS})

target_link_libraries(invert ${OpenCV_LIBS})
