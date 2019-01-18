#!/bin/sh
# Without rendering
#bazel run :python_random_agent --        --length=10000
# With rendering
bazel run :python_random_agent --define graphics=sdl --        --length=10000 --width=640 --height=480
