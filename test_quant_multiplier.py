#!/usr/bin/env python

import quantization
import model
import numpy as np
from sys import argv

m = model.TFLiteModel(argv[1])

for op in m:
    if op.opname not in ('CONV_2D', 'DEPTHWISE_CONV_2D'):
        continue
    zps = [i.scale for i in m.get_inputs_for_op(op)]
    n, qm = quantization.compute_multiplier(*zps)
    print 'n=%s, qm=%s' % (n, qm)
    print '%s ?= %s' % (
        quantization.quantized_multiplier_mult(1, qm, n),
        (zps[0]*zps[1])/zps[2])
