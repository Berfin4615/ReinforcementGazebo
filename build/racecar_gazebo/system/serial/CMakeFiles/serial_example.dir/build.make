# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python3.8/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.8/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/racecar_rl/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/racecar_rl/build

# Include any dependencies generated for this target.
include racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/compiler_depend.make

# Include the progress variables for this target.
include racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/progress.make

# Include the compile flags for this target's objects.
include racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/flags.make

racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/examples/serial_example.cc.o: racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/flags.make
racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/examples/serial_example.cc.o: /root/racecar_rl/src/racecar_gazebo/system/serial/examples/serial_example.cc
racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/examples/serial_example.cc.o: racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/root/racecar_rl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/examples/serial_example.cc.o"
	cd /root/racecar_rl/build/racecar_gazebo/system/serial && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/examples/serial_example.cc.o -MF CMakeFiles/serial_example.dir/examples/serial_example.cc.o.d -o CMakeFiles/serial_example.dir/examples/serial_example.cc.o -c /root/racecar_rl/src/racecar_gazebo/system/serial/examples/serial_example.cc

racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/examples/serial_example.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/serial_example.dir/examples/serial_example.cc.i"
	cd /root/racecar_rl/build/racecar_gazebo/system/serial && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/racecar_rl/src/racecar_gazebo/system/serial/examples/serial_example.cc > CMakeFiles/serial_example.dir/examples/serial_example.cc.i

racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/examples/serial_example.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/serial_example.dir/examples/serial_example.cc.s"
	cd /root/racecar_rl/build/racecar_gazebo/system/serial && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/racecar_rl/src/racecar_gazebo/system/serial/examples/serial_example.cc -o CMakeFiles/serial_example.dir/examples/serial_example.cc.s

# Object files for target serial_example
serial_example_OBJECTS = \
"CMakeFiles/serial_example.dir/examples/serial_example.cc.o"

# External object files for target serial_example
serial_example_EXTERNAL_OBJECTS =

/root/racecar_rl/devel/lib/serial/serial_example: racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/examples/serial_example.cc.o
/root/racecar_rl/devel/lib/serial/serial_example: racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/build.make
/root/racecar_rl/devel/lib/serial/serial_example: /root/racecar_rl/devel/lib/libserial.so
/root/racecar_rl/devel/lib/serial/serial_example: racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/root/racecar_rl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /root/racecar_rl/devel/lib/serial/serial_example"
	cd /root/racecar_rl/build/racecar_gazebo/system/serial && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/serial_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/build: /root/racecar_rl/devel/lib/serial/serial_example
.PHONY : racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/build

racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/clean:
	cd /root/racecar_rl/build/racecar_gazebo/system/serial && $(CMAKE_COMMAND) -P CMakeFiles/serial_example.dir/cmake_clean.cmake
.PHONY : racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/clean

racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/depend:
	cd /root/racecar_rl/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/racecar_rl/src /root/racecar_rl/src/racecar_gazebo/system/serial /root/racecar_rl/build /root/racecar_rl/build/racecar_gazebo/system/serial /root/racecar_rl/build/racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : racecar_gazebo/system/serial/CMakeFiles/serial_example.dir/depend

