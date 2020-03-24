# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lancelot/Desktop/matrix_invert

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lancelot/Desktop/matrix_invert/build

# Include any dependencies generated for this target.
include CMakeFiles/invert_benchmark.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/invert_benchmark.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/invert_benchmark.dir/flags.make

CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.o: CMakeFiles/invert_benchmark.dir/flags.make
CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.o: ../src/invert_benchmark.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lancelot/Desktop/matrix_invert/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.o -c /home/lancelot/Desktop/matrix_invert/src/invert_benchmark.cpp

CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lancelot/Desktop/matrix_invert/src/invert_benchmark.cpp > CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.i

CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lancelot/Desktop/matrix_invert/src/invert_benchmark.cpp -o CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.s

CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.o.requires:

.PHONY : CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.o.requires

CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.o.provides: CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.o.requires
	$(MAKE) -f CMakeFiles/invert_benchmark.dir/build.make CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.o.provides.build
.PHONY : CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.o.provides

CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.o.provides.build: CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.o


# Object files for target invert_benchmark
invert_benchmark_OBJECTS = \
"CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.o"

# External object files for target invert_benchmark
invert_benchmark_EXTERNAL_OBJECTS =

invert_benchmark: CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.o
invert_benchmark: CMakeFiles/invert_benchmark.dir/build.make
invert_benchmark: src/invert/libinvert.a
invert_benchmark: /usr/local/lib/libbenchmark.a
invert_benchmark: /usr/local/lib/libopencv_dnn.so.3.4.9
invert_benchmark: /usr/local/lib/libopencv_highgui.so.3.4.9
invert_benchmark: /usr/local/lib/libopencv_ml.so.3.4.9
invert_benchmark: /usr/local/lib/libopencv_objdetect.so.3.4.9
invert_benchmark: /usr/local/lib/libopencv_shape.so.3.4.9
invert_benchmark: /usr/local/lib/libopencv_stitching.so.3.4.9
invert_benchmark: /usr/local/lib/libopencv_superres.so.3.4.9
invert_benchmark: /usr/local/lib/libopencv_videostab.so.3.4.9
invert_benchmark: /usr/local/lib/libopencv_calib3d.so.3.4.9
invert_benchmark: /usr/local/lib/libopencv_features2d.so.3.4.9
invert_benchmark: /usr/local/lib/libopencv_flann.so.3.4.9
invert_benchmark: /usr/local/lib/libopencv_photo.so.3.4.9
invert_benchmark: /usr/local/lib/libopencv_video.so.3.4.9
invert_benchmark: /usr/local/lib/libopencv_videoio.so.3.4.9
invert_benchmark: /usr/local/lib/libopencv_imgcodecs.so.3.4.9
invert_benchmark: /usr/local/lib/libopencv_imgproc.so.3.4.9
invert_benchmark: /usr/local/lib/libopencv_core.so.3.4.9
invert_benchmark: /usr/lib/x86_64-linux-gnu/librt.so
invert_benchmark: CMakeFiles/invert_benchmark.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lancelot/Desktop/matrix_invert/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable invert_benchmark"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/invert_benchmark.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/invert_benchmark.dir/build: invert_benchmark

.PHONY : CMakeFiles/invert_benchmark.dir/build

CMakeFiles/invert_benchmark.dir/requires: CMakeFiles/invert_benchmark.dir/src/invert_benchmark.cpp.o.requires

.PHONY : CMakeFiles/invert_benchmark.dir/requires

CMakeFiles/invert_benchmark.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/invert_benchmark.dir/cmake_clean.cmake
.PHONY : CMakeFiles/invert_benchmark.dir/clean

CMakeFiles/invert_benchmark.dir/depend:
	cd /home/lancelot/Desktop/matrix_invert/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lancelot/Desktop/matrix_invert /home/lancelot/Desktop/matrix_invert /home/lancelot/Desktop/matrix_invert/build /home/lancelot/Desktop/matrix_invert/build /home/lancelot/Desktop/matrix_invert/build/CMakeFiles/invert_benchmark.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/invert_benchmark.dir/depend
