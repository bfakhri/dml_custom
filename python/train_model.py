from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import random
import numpy as np

import deepmind_lab
import tensorflow as tf

import sys
print('PYTHON VERSION - ', sys.version)

# For the DML random agent dataset
import random_dataset
# For the model that we will train
import model

ds = random_dataset.dml_dataset()
model = model.Model(ds.shape)

for i in range(100):
    batch = ds.get_batch()
    model.train_step(batch)
