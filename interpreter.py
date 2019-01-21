import numpy as np
from model import TFLiteModel
import quantization

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


def determine_padding_dims(padding, filter_shape, input_shape):
    # compute height and width needed for padding according to `padding` (either
    # SAME or VALID), where `input_shape` is the shape of the input and
    # `filter_shape` is the shape of the kernel.
    pass


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


def offset(shape, i0, i1, i2, i3):
    return ((i0 * shape[1] + i1) * shape[2] + i2) * shape[3] + i3


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

    # batches, input_height, input_width, input_channels = data.shape
    # filter_height, filter_width, _, output_channels = weights.shape
    # _, output_height, output_width, _ = output.shape
    # stride_h, stride_w = op.stride
    # dilation_h, dilation_w = op.dilation_factor

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
    # for b in range(batches):
    #     for out_y in range(output_height):
    #         for out_x in range(output_width):
    #             for out_c in range(output_channels):
    #                 # TODO: figure out padding
    #                 in_x_origin = (out_x * stride_w)
    #                 in_y_origin = (out_y * stride_h)
    #                 int32_acc = int32(0)
    #                 for filter_y in range(filter_height):
    #                     for filter_x in range(filter_width):
    #                         for in_c in range(input_channels):
    #                             in_x = in_x_origin + dilation_w * filter_x
    #                             in_y = in_y_origin + dilation_h * filter_y
    #                             if (0 <= in_x < input_w) and (0 <= in_y < input_h):
    #                                 pass
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

    weights, bias, input_data = split_conv2d_inputs(inputs)

    num_batches, input_h, input_w, input_c = input_data.shape
    filter_h, filter_w, _, output_c = weights.shape
    _, output_h, output_w, _ = output.shape
    stride_h, stride_w = op.stride
    dilation_h, dilation_w = op.dilation_factor
    depth_multiplier = op.depth_multiplier

    filter_s = weights.scale
    filter_z = weights.zero_point
    input_s = input_data.scale
    input_z = input_data.zero_point
    output_s = output.scale
    output_z = output.zero_point

    # padding is a function of the shape of the input and outptu
    pad_h, pad_w = determine_padding_dims(
        op.padding, weights.shape, input_data.shape)

    # compute the shift and multiplier once and for all here.
    n, qm = quantization.compute_multiplier(filter_s, input_s, output_s)

    print 'Computing Depthwise Conv2D'

    for b in range(num_batches):
        for out_y in range(output_h):
            for out_x in range(output_w):
                for ic in range(input_c):
                    for m in range(depth_multiplier):
                        oc = m + ic * depth_multiplier
                        # pad width and height here
                        in_x_origin = (out_x * stride_w) - 0
                        in_y_origin = (out_y * stride_h) - 0
                        acc = int32(0)
                        for filter_y in range(filter_h):
                            for filter_x in range(filter_w):
                                in_x = in_x_origin + dilation_w * filter_x
                                in_y = in_y_origin + dilation_h * filter_y

                                if (0 <= in_x < input_w and 0 <= in_y < input_h):
                                    iv = input_data[offset(
                                        input_data.shape,
                                        b, in_y, in_x, ic
                                    )]
                                    fv = weights[offset(
                                        weights.shape,
                                        0, filter_y, filter_x, oc
                                    )]
                                    acc += int32((fv + filter_s) * (iv + input_s))
                            # end filter_x
                        # end filter_y
                        acc += int32(bias[oc])

                        # multiply by M (which we have as the pair (n, qm)). We
                        # want to get readings that are accurate, so we cannot
                        # simply multiply the original M here. We have to go
                        # through all the hoops that gemmlowp also does.
                        acc = quantized_multiplier_mult(acc, qm, n)

                    # end m
                # end ic
            # end out_x
        # end out_y
    # end b

    #


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

        opname = op.opname
        if 'CONV_2D' == opname:
            conv2d(op, op_inputs, op_outputs)
        elif 'DEPTHWISE_CONV_2D' == opname:
            depthwise_conv2d(op, op_inputs, op_outputs)
        elif 'ADD' == opname:
            add(op, op_inputs, op_outputs)
        elif 'AVERAGE_POOL_2D' == opname:
            avgpool2d(op, op_inputs, op_outputs)
        elif 'RESIZE_BILINEAR' == opname:
            resize_bilinear(op, op_inputs, op_outputs)
        else:
            print 'Unknown operator: %s' % (opname,)
            return

    print 'Done. Prepping output'
