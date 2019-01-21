#!/usr/bin/env python

import quantization
import model
import random
import numpy as np
from interpreter import is_bias_tensor, is_weights_tensor
from sys import argv

m = model.TFLiteModel(argv[1], parse_data=False)

for op in m:
    if op.opname not in ('CONV_2D', 'DEPTHWISE_CONV_2D'):
        continue
    ins = [m.tensors[idx] for idx in op.inputs if
           is_bias_tensor(m.tensors[idx]) or is_weights_tensor(m.tensors[idx])]
    outs = [i.scale for i in m.get_outputs_for_op(op)]
    zps = [i.scale for i in ins] + outs
    n, qm = quantization.compute_multiplier(*zps)
    large_num = pow(-1,random.randint(0,1)) * (1 << 25)
    print '%s ?= %s (n=%s, qm=%s)' % (
        quantization.quantized_multiplier_mult(large_num, qm, n),
        large_num*((zps[0]*zps[1])/zps[2]), n, qm)
