'''
Created on Dec 18, 2019

@author: rkaria
'''

import random

import numpy as np
import tensorflow as tf

# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
# The link above gives some info about keras.
#
# Exact reproducibility is difficult since we rely on FF as well.


def set_random_seeds():

    random.seed(0xDEADC0DE)
    tf.compat.v1.random.set_random_seed(0xBADC0DE)
    np.random.seed(0xABCDEF)
