import numpy as np
from model import TFLiteModel

float32 = np.float32
int32 = np.int32
uint8 = np.uint8

# "Toy" interpreter that doesn't interpret anything. It's purpose is only to
# illustrate how `data` travels through `model`
def run_interactive_no_eval(model_path, data):
    model = TFLiteModel(model_path, parse_data=False)

    assert data is not None, 'input data cannot be none'

    # don't reshape data as it might just be a string. Besides, we don't
    # evaluate anything so it doesn't matter anyway.
    model.set_input(data, reshape=False)

    i = 0

    for op in model:
        print op

        # we run through the inputs of `op` and find the entry with `data`. This
        # is then copied into the outputs of `op`. We also ad a '.' for each
        # "evaluation".
        data1 = None
        for idx in op.inputs:
            if model.tensors[idx].data is not None:
                print model.tensors[idx].data
                data1 = model.tensors[idx].data + ' ' + str(i)
        if data1 is None:
            print 'Disconnected graph, oh no!'
            return
        i += 1
        for idx in op.outputs:
            if model.tensors[idx].data is not None:
                print 'Someone already wrote to the tensor at %s' % (idx,)
                return
            else:
                model.tensors[idx].data = data1

    print model.get_output().data

# def get_quantization_params(tensor, verbose=False):
#     zero_point = uint8(tensor['zero_point'])
#     scale = float32(tensor['scale'])
#     if verbose:
#         print 'Quantization Params for %s: Z=%s, S=%s' % (tensor['type'], zero_point, scale)
#     return zero_point, scale

# def split_conv2d_inputs(inputs):
#     assert len(inputs) == 3, "Did not get three (weights, bias, input) inputs for Conv2d"
#     bias = None
#     weights = None
#     data = None
#     for i in range(len(inputs)):
#         tensor = inputs[i]
#         if ('Conv2D_Fold_bias' in tensor['type'] or
#             'Conv2D_bias' in tensor['type'] or
#             'depthwise_Fold_bias' in tensor['type']):
#             bias = tensor
#         elif 'weights_quant' in tensor['type']:
#             weights = tensor
#         else:
#             data = tensor
#     if weights is None or bias is None or data is None:
#         print 'Could not extract approriate inputs for operator'
#         print 'weights=%s, bias=%s, data=%s' % (weights, bias, data)
#         return None
#     return weights, bias, data

def conv2d(op, inputs, output):
    print 'Computing CONV2D'

def depthwise_conv2d(op, inputs, output):
    print 'Computing Depthwise Conv2D'

def add(op, inputs, output):
    print 'Computing Residual'

def avgpool2d(op, inputs, output):
    print 'Computing Average Pool2D'

def resize_bilinear(op, inputs, output):
    print 'Computing Resize Bilinear'

def run(model, input_data):
    assert input_data is not None, 'Input data cannot be None'
    model.set_input(input_data)

    for op in model:
        op_inputs = [model.tensors[idx] for idx in op.inputs]
        op_outputs = [model.tensors[idx] for idx in op.outputs]

        # assume all operations have one output only
        output_tensor = op_outputs[0]

        for op in model:
            opname = op.opname
            if 'CONV_2D' == opname:
                output = conv2d(op, op_inputs, output_tensor)
            elif 'DEPTHWISE_CONV_2D' == opname:
                output = depthwise_conv2d(op, op_inputs, output_tensor)
            elif 'ADD' == opname:
                output = add(op, op_inputs, output_tensor)
            elif 'AVERAGE_POOL_2D' == opname:
                output = avgpool2d(op, op_inputs, output_tensor)
            elif 'RESIZE_BILINEAR' == opname:
                output = resize_bilinear(op, op_inputs, output_tensor)
            else:
                print 'Unknown operator: %s' % (opname,)
                return

    print 'Done. Prepping output'
