import numpy as np

# "Toy" interpreter that doesn't interpret anything. The purpose is to
# illustrate how `data' is passed through the graph.
def run_interactive_no_eval(model, data):
    assert data is not None, 'input data cannot be none'
    tensors = model.get_tensors()
    operators = model.get_operators()
    model.tensors[model.graph.Inputs(0)]['data'] = data
    input_layer = model.get_input()

    for op_idx, op in model:
        # print operator + inputs and outputs before "evaluation"
        model.print_operator(op_idx)
        print 'Computing %s ... (though not really)' % (op['type'], )
        data = list()
        output_data = None
        for in_idx in op['inputs']:
            data.append(tensors[in_idx]['data'])
            print data

        # flatten, since there's going to be a bunch of None's here
        for x in data:
            if x is not None:
                output_data = x
                break

        # ensure that we actually found something
        assert output_data is not None, 'Disconnected graph'
        # set result of "evaluation".
        model.set_output_on_operator(op_idx, output_data)

        # print operator after evaluation
        model.print_operator(op_idx)

        print '\n--------- ... ------------\n'
        raw_input()


def split_conv2d_inputs(inputs):
    assert len(inputs) == 3, "Did not get three (weights, bias, input) inputs for Conv2d"
    bias = None
    weights = None
    data = None
    for i in range(len(inputs)):
        tensor = inputs[i]
        if ('Conv2D_Fold_bias' in tensor['type'] or
            'Conv2D_bias' in tensor['type'] or
            'depthwise_Fold_bias' in tensor['type']):
            bias = tensor
        elif 'weights_quant' in tensor['type']:
            weights = tensor
        else:
            data = tensor
    if weights is None or bias is None or data is None:
        print 'Could not extract approriate inputs for operator'
        print 'weights=%s, bias=%s, data=%s' % (weights, bias, data)
        return None
    return weights, bias, data

def conv2d(op, inputs):
    params = split_conv2d_inputs(inputs)
    if params is None:
        print 'Could not compute %s' % (op,)
        return
    weights, bias, data = params

    print 'Computing CONV2D'

def depthwise_conv2d(op, inputs):
    params = split_conv2d_inputs(inputs)
    if params is None:
        print 'Could not compute %s' % (op,)
        return
    weights, bias, data = params

    print 'Computing Depthwise Conv2D'

def add(op, inputs):
    print 'Computing Residual'

def avgpool2d(op, inputs):
    print 'Computing Average Pool2D'

def resize_bilinear(op, inputs):
    print 'Computing Resize Bilinear'

def run(model, input_data):

    assert input_data is not None, 'Input data cannot be None'

    tensors = model.get_tensors()
    operators = model.get_operators()
    model.set_input(input_data)

    print 'Setting input'

    for op_idx, op in model:
        inputs = [tensors[i] for i in op['inputs']]

        if op['type'] == 'CONV_2D':
            output = conv2d(op, inputs)
        elif op['type'] == 'DEPTHWISE_CONV_2D':
            output = depthwise_conv2d(op, inputs)
        elif op['type'] == 'ADD':
            output = add(op, inputs)
        elif op['type'] == 'AVERAGE_POOL_2D':
            output = avgpool2d(op, inputs)
        elif op['type'] == 'RESIZE_BILINEAR':
            output = resize_bilinear(op, inputs)
        else:
            print 'unknown operator type: %s' % (op['type'],)

    print 'Done. Prepping output'
