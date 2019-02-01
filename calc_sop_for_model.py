#!/usr/bin/env python

from model import TFLiteModel
import sys

if len(sys.argv) < 2:
    print 'usage: %s [model_path]' % (sys.argv[0], )
    exit(0)

model = TFLiteModel(sys.argv[1], parse_data=False)

conv2d_sops = list()
conv2d_lengths = list()
dwconv2d_sops = list()
dwconv2d_lengths = list()

for op in model:
    if op.opname not in ('CONV_2D', 'DEPTHWISE_CONV_2D'):
        continue
    _, oh, ow, out_c = list(model.get_outputs_for_op(op))[0].shape

    _, wh, ww, in_c = model.get_named_inputs_for_op(op)['weights'][0].shape
    if op.opname == 'CONV_2D':
        # longest vector in a dot product here is height*width*in_channels
        conv2d_lengths.append(wh*ww*in_c)
        # number of sum-of-products for this operation
        conv2d_sops.append(oh*ow*out_c)
    else:
        # depthwise conv2d
        dwconv2d_lengths.append(wh*wh)
        dwconv2d_sops.append(oh*ow*out_c)

conv2d_sops = sorted(list(set(conv2d_sops)), reverse=True)
dwconv2d_sops = sorted(list(set(dwconv2d_sops)), reverse=True)
conv2d_lengths = sorted(list(set(conv2d_lengths)), reverse=True)
dwconv2d_lengths = sorted(list(set(dwconv2d_lengths)), reverse=True)

print 'Model:', sys.argv[1]
print '--------------------------------------------'
print 'sum-of-products (conv2d):   ', ', '.join(str(_) for _ in conv2d_sops[:5])
print 'longest vector (conv2d):    ', ', '.join(str(_) for _ in conv2d_lengths[:5])
print ' .. '
print 'sum-of-products (dwconv2d): ', ', '.join(str(_) for _ in dwconv2d_sops[:5])
print 'longest vector (dwconv2d):  ', ', '.join(str(_) for _ in dwconv2d_lengths[:5])
print ' .. '
