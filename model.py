# Parse a .tflite file and return the content as a list of layers. I.e., if the
# .tflite describes a network with the structure
#
#  input -> FC -> FC -> output
#
# then this script outputs a json compliant list of four elements where each
# element corresponds to a layer in the network. A 'layer' in the output is
# guaranteed to have, at least, the key 'type' which denotes its type (e.g., FC,
# Conv2D, input).
#
# We only care about models with 1 subgraph and which has 1 input and 1 output.

import numpy as np
from tflite.Model import Model
from tflite.BuiltinOperator import BuiltinOperator

# import the relevant option classes that is supported
from tflite.Conv2DOptions import Conv2DOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from tflite.ResizeBilinearOptions import ResizeBilinearOptions


def load_model(model_path):
    with open(model_path, 'rb') as f:
        buf = f.read()
    buf = bytearray(buf)
    return Model.GetRootAsModel(buf, 0)


def load_opcodes(model):
    opcodes = []
    ops = [(getattr(BuiltinOperator, op), op)
           for op in dir(BuiltinOperator) if not op.startswith('__')]
    ops.sort()
    for i in range(model.OperatorCodesLength()):
        op = ops[model.OperatorCodes(i).BuiltinCode()]
        opcodes.append(op[1])
    return opcodes


class TFLiteModel:

    padding_schemes = ['SAME', 'VALID']
    # these are the only types we care to support. Note that we don't parse the
    # scale (which is FLOAT32) since it's not stored in a Buffer
    data_types = [None, None, 'INT32', 'UINT8']

    def __init__(self, model_path, delay_load=False, parse_data=True):
        self.model_path = model_path
        self.parse_data = parse_data
        # load model
        model = load_model(model_path)
        assert model is not None, 'Could not load model'
        # find names of all the valid opcodes of this model
        opcodes = load_opcodes(model)
        self.model = model
        self.opcodes = opcodes
        self.graph = self.model.Subgraphs(0)

        # Keep track of two lists: One with operations and one with tensors.
        self.operators = []
        self.tensors = []

        self.__current_op_idx = 0

        if not delay_load:
            self.load()

    def get_operators(self):
        return self.operators

    def get_tensors(self):
        return self.tensors

    def load(self):
        num_ops = self.graph.OperatorsLength()
        for i in range(num_ops):
            op = self.graph.Operators(i)
            self.operators.append(self.__parse_op(op))

        num_tensors = self.graph.TensorsLength()
        for i in range(num_tensors):
            tensor = self.graph.Tensors(i)
            self.tensors.append(self.__parse_tensor(tensor))

    def get_input(self):
        # assume only one input
        idx = self.graph.Inputs(0)
        return self.tensors[idx]

    def set_input(self, data):
        ix = self.get_input()
        if data.shape != ix['shape']:
            data = data.reshape(ix['shape'])
        ix['data'] = data

    def get_output(self):
        # ditto
        idx = self.graph.Outputs(0)
        return self.tensors[idx]

    def set_output_on_operator(self, op_idx, data):
        if op_idx >= len(self.operators):
            print 'No operator with index %s' % (op_idx,)
        for output_idx in self.operators[op_idx]['outputs']:
            self.tensors[output_idx]['data'] = data

    def print_operator(self, idx):
        if idx >= len(self.operators):
            print 'No operator at index: %s' % (idx,)
        op = self.operators[idx]
        print 'Operator Nr=%s: %s' % (idx, op['type'])
        print 'Content=%s' % (op,)
        print 'Input tensors:'
        for tensor_idx in op['inputs']:
            print '(idx=%s) %s' % (tensor_idx, self.tensors[tensor_idx])
        print '\nOutput tensors:'
        for tensor_idx in op['outputs']:
            print '(idx=%s) %s' % (tensor_idx, self.tensors[tensor_idx])
        print ''

    def __parse_op(self, op):
        ins = [idx for idx in op.InputsAsNumpy()]
        outs = [idx for idx in op.OutputsAsNumpy()]
        operator = {
            'type': self.opcodes[op.OpcodeIndex()],
            'inputs': ins,
            'outputs': outs
        }
        typ = operator['type']
        # no options for typ == 'ADD' or typ == 'AVERAGE_POOL_2D'
        options = op.BuiltinOptions()
        if typ == 'CONV_2D':
            o = Conv2DOptions()
            o.Init(options.Bytes, options.Pos)
            operator['stride'] = (o.StrideH(), o.StrideW())
            operator['padding'] = self.padding_schemes[o.Padding()]
        elif typ == 'DEPTHWISE_CONV_2D':
            o = DepthwiseConv2DOptions()
            o.Init(options.Bytes, options.Pos)
            operator['stride'] = (o.StrideH(), o.StrideW())
            operator['padding'] = self.padding_schemes[o.Padding()]
            operator['depth_multiplier'] = o.DepthMultiplier()
        elif typ == 'RESIZE_BILINEAR':
            o = ResizeBilinearOptions()
            o.Init(options.Bytes, options.Pos)
            operator['align_corners'] = o.AlignCorners()
        return operator

    def __cast_buf(self, buf, buf_typ):
        # convert `buf` of uint8 to buf_typ (currently only support int32). Byte
        # order is assumed to be little endian.
        bytelen = None
        np_typ = None
        if buf_typ == 'INT32':
            np_typ = np.dtype('int32')
            bytelen = 4
        else:
            raise ValueError('Unknown data type: %s' % (buf_typ,))

        assert len(buf) % bytelen == 0, 'Buffer not a multiple of type size'
        # assume little endian order
        buf1 = list()
        i = 0
        while i < len(buf):
            x = 0
            for j in range(bytelen):
                x |= buf[i + j] << (8 * j)
            i += bytelen
            buf1.append(x)
        return np.asarray(buf1, dtype=np_typ)

    def __parse_tensor(self, tensor):
        quant = tensor.Quantization()
        scale = quant.ScaleAsNumpy()
        zero_point = quant.ZeroPointAsNumpy()
        if type(scale) == np.ndarray:
            scale = scale[0]
        if type(zero_point) == np.ndarray:
            zero_point = zero_point[0]
        t = {'type': tensor.Name(),
             'scale': scale,
             'zero_point': zero_point,
             'shape': tuple(tensor.ShapeAsNumpy())}
        # next up is parsing any potential data that this tensor has. If
        # `parse_data` is false, then we don't do any parsing and just set the
        # data to None. This can be useful for debugging.
        if not self.parse_data:
            t['data'] = None
            return t
        buf_idx = tensor.Buffer()
        buf_typ = self.data_types[tensor.Type()]
        assert buf_typ is not None, 'Unknown data type: %s' % (tensor.Type(),)
        buf = self.model.Buffers(buf_idx)
        buf = buf.DataAsNumpy()
        if type(buf) != np.ndarray:
            # buffer is just a scalar
            t['data'] = buf
            return t
        if buf_typ != 'UINT8':
            buf = self.__cast_buf(buf, buf_typ)
        try:
            buf = buf.reshape(t['shape'])
            t['data'] = buf
        except:
            print 'Could not reshape %s to %s' % (t['type'], t['shape'])
            # welp, buf was probably a scalar or something
            t['data'] = buf
        return t

    def __repr__(self):
        return 'Version %s (%s) opcodes=%s' % (
            self.model.Version(),
            self.model_path,
            self.opcodes)

    def __iter__(self):
        return self

    def next(self):
        if self.__current_op_idx >= len(self.operators):
            raise StopIteration
        else:
            op_idx = self.__current_op_idx
            self.__current_op_idx += 1
            return op_idx, self.operators[op_idx]



# For illustration purposes:
def step_through_model(model_path):
    model = TFLiteModel(model_path, parse_data=False)

    tensors = model.get_tensors()

    input_layer = model.get_input()
    output_layer = model.get_output()

    print input_layer
    try:
        for idx, op in enumerate(model.get_operators()):
            print 'Operator %s: %s (%s)' % (idx, op['type'], op)
            inputs_idx = op['inputs']
            print 'input tensors:'
            for i in inputs_idx:
                print 'Tensors_idx=%s: %s' % (i, tensors[i])
            print ''
            outputs_idx = op['outputs']
            print 'output tensors:'
            for i in outputs_idx:
                print 'Tensors_idx=%s: %s' % (i, tensors[i])
            print ''
            _ = raw_input('...')
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    from sys import argv

    if len(argv) < 2:
        print 'Usage: %s [model_path]' % (argv[0],)
        exit(0)

    step_through_model(argv[1])
