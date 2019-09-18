#!/usr/bin/env python

from interpreter_v3 import load_image_data
from model import TFLiteModel
from PIL import Image

import sys

model = TFLiteModel(sys.argv[1])
input_shape = model.get_input().shape
data = load_image_data(sys.argv[2], input_shape, use_keras=True)

f = open(sys.argv[3], 'w')
for x in data.flatten():
    print >>f, int(x)
