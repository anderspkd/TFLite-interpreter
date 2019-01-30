#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:00:38 2019

@author: deescuderoo
"""

import quantization
import numpy as np

n_iter = 100000

fl = np.random.rand(n_iter)
x = np.int32(np.random.randint(quantization.INT32_MIN, quantization.INT32_MAX, (n_iter,)))

real = x * fl

p_numpy = []
p_custom = []

for i in range(n_iter):        
    n, qm = quantization.compute_multiplier(fl[i])
    
    p_numpy.append(quantization.quantized_multiplier_mult(x[i], qm, n, True, True))
    p_custom.append(quantization.quantized_multiplier_mult(x[i], qm, n, True, False))

p_numpy = np.array(p_numpy)
p_custom = np.array(p_custom)

print np.mean(np.abs(p_custom - real))
print np.mean(np.abs(p_numpy - real))