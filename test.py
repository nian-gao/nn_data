from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 仅用cpu

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
