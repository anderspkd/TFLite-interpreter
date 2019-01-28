# coding: utf-8

import model
m = model.TFLiteModel('data/mobilenet_v2_1.0_224_quant.tflite')
for op in m:
    if op.opname == 'CONV_2D':
        inputs = m.get_named_inputs_for_op(op)
        outputs = list(m.get_outputs_for_op(op))
        weights_shape = inputs['weights'][0].shape
        if weights_shape[1] == 1 and weights_shape[2] == 1:
            print 'shapes:',inputs['_'][0].shape, inputs['weights'][0].shape, outputs[0].shape
            print 'matrix size: %s x %s = %s' % (inputs['_'][0].shape[1], inputs['_'][0].shape[3], inputs['_'][0].shape[1] * inputs['_'][0].shape[3])
            print 'num matmults', outputs[0].shape[2] * outputs[0].shape[3]
            print '..'
            
            
