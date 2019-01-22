#!/usr/bin/env python

from model import Tensor
import numpy as np

def offset(shape, i0, i1, i2, i3):
    return ((i0 * shape[1] + i1) * shape[2] + i2) * shape[3] + i3

int32 = np.int32

output_w, output_h = (1, 1)
input_w, input_h = (7, 7)
input_c = 1280
padding_w, padding_h = (0, 0)
stride_w, stride_h = (1, 1)
output_c = 1280
filter_w, filter_h = (7, 7)

in_flat_shape = 1 * 7 * 7 * 1280
in_shape = (1, 7, 7, 1280)
out_flat_shape = 1 * 1 * 1 * 1280
out_shape = (1, 1, 1, 1280)

# create a flat tensor with shape `in_shape`
input_tensor = np.random.randint(
    0, 255, size=in_shape, dtype='uint8'
)
output_tensor = np.zeros(out_shape)

# this is the simplest non-general setting which assumes the filter size and
# input sizes are the same. Thus it suffices to iterate over the channels and
# then compute an average of
for c in range(input_c):
    acc = int32(0)
    fc = 0
    for x in range(filter_w):
        for y in range(filter_h):
            # coincidentally, the filter size and input size are the same.
            acc += input_tensor[0][x][y][c]
            fc += 1
    if acc > 0:
        avg = (acc + fc / 2) / fc
    else:
        avg = (acc - fc / 2) / fc
    avg = max(avg, 0)
    avg = min(avg, 255)
    output_tensor[0][0][0][c] = avg
    print 'fc = %s, acc = %s, avg = %s' % (fc, acc, avg)


print output_tensor
