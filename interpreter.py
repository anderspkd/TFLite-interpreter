import numpy as np
from model import TFLiteModel

# This "interpreter" is purely for illustrative purposes. It goes through the
# entire model, passing the input to the output at each layer. It serves to
# illustrate the flow of the input through the model.
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


# this is sorta hacky. But I'm not sure how else to determine where a specific
# tensor is used.
def is_bias_tensor(tensor):
    n = tensor.name
    return ('Conv2D_Fold_bias' in n or
            'Conv2D_bias' in n or
            'depthwise_Fold_bias' in n)


def is_weights_tensor(tensor):
    n = tensor.name
    return 'weights_quant' in n


def split_conv2d_inputs(inputs):
    assert len(inputs) == 3
    bias = None
    weights = None
    data = None
    for i in range(3):
        tensor = inputs[i]
        if is_bias_tensor(tensor):
            bias = tensor
            continue
        if is_weights_tensor(tensor):
            weights = tensor
            continue
        data = tensor
    # make sure we fail if we could not find suitable data
    assert data is not None
    assert weights is not None
    assert bias is not None
    return weights, bias, data


float32 = np.float32
int32 = np.int32
uint8 = np.uint8


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
        assert len(op_outputs) == 1, 'Cannot handle more than 1 output'
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
