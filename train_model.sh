#!/bin/sh
# With rendering
bazel run :python_train_model --define graphics=sdl -- 
