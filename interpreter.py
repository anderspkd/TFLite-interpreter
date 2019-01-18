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

        raw_input()
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
    # The implementation here follows the one in reference_ops.h. See
    # /tensorflow/lite/kernels/internal/reference/reference_ops.h#L321
    weights, bias, data = split_conv2d_inputs(inputs)

    # data shape. Output is assumed to have the same shape
    #  [batch, height, width, channels]
    # weights shape
    #  [filter_height, filter_width, in_channels, out_channels]
    # bias is a vector with filter_height entries

    print 'Computing Conv2D'

    batches, input_height, input_width, input_channels = data.shape
    filter_height, filter_width, _, output_channels = weights.shape
    _, output_height, output_width, _ = output.shape
    stride_h, strid_w = op.stride
    dilation_h, dilation_w = op.dilation_factor

    # filter_offset is weight quantization
    # input_offset is likewise input quantization
    # ouput_offset, ditto

    # Offset(...) is defined at
    # lite/kernels/internal/types.h:372

    # Flatten input_data first? That would make it closer to the functions in
    # reference_ops.h. It might also make it easier to work with...

    # can probably make an assumption about the number of batches in the
    # input. In particular that there's only ever going to be one.

    # lmao this is slow as fuck
    for b in range(batches):
        for out_y in range(output_height):
            for out_x in range(output_width):
                for out_c in range(output_channels):
                    # TODO: figure out padding
                    in_x_origin = (out_x * stride_w)
                    in_y_origin = (out_y * stride_h)
                    int32_acc = int32(0)
                    for filter_y in range(filter_height):
                        for filter_x in range(filter_width):
                            for in_c in range(input_channels):
                                in_y_origin + dilation_h * filter_y
                                if (0 <= in_x < input_w) and (0 <= in_y < input_h):
                                    pass
                            # end in_c
                        # end filter_x
                    # end filter_Y

                    # add bias here, perform quantization adjustment and save
                    # output.
                # end out_c
            # end out_x
        # end out_y
    # end b


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
