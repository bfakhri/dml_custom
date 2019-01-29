#!/bin/sh
# With rendering
#bazel run :python_random_dataset --define graphics=sdl --   
## With rendering and custom dimensions
#bazel run :python_random_dataset --        --length=10000 --width=40 --height=30
bazel run :python_random_dataset --
