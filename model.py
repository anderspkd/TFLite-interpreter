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
    # Load a model from a .tflite file at `model_path`
    with open(model_path, 'rb') as f:
        buf = f.read()
    buf = bytearray(buf)
    return Model.GetRootAsModel(buf, 0)


def load_opcodes(model):
    # Given a flatbuffer model, extract names of all operators that are used in
    # `model`.
    opcodes = []
    ops = [(getattr(BuiltinOperator, op), op)
           for op in dir(BuiltinOperator) if not op.startswith('__')]
    ops.sort()
    for i in range(model.OperatorCodesLength()):
        op = ops[model.OperatorCodes(i).BuiltinCode()]
        opcodes.append(op[1])
    return opcodes


class InvalidOperator(Exception):
    pass


class Operator:

    # Supported keys:
    #
    #  name: A user chosen name. Defaults to `None`
    #  opname: Name of the operator (e.g., CONV_2D)
    #  inputs: An array of Tensor indices pointing to inputs
    #  outputs: Ditto, but for outputs
    #
    # For additional options that depend on the operator type (i.e., `opname`),
    # see the different subclasses (e.g., Conv2DOperator).

    valid_opcodes = []

    def __init__(self, flatbuf_op, name=None):
        self._flatbuf_op = flatbuf_op
        self._flatbuf_options_obj = None
        self._supported_options = []
        op_idx = flatbuf_op.OpcodeIndex()
        if op_idx >= len(self.valid_opcodes):
            raise InvalidOperator('Unsupported operator: %s' % (op_idx,))
        # gather inputs and outputs
        self.inputs = [idx for idx in flatbuf_op.InputsAsNumpy()]
        self.outputs = [idx for idx in flatbuf_op.OutputsAsNumpy()]
        # set name
        self.opname = self.valid_opcodes[op_idx]
        self.name = name
        # load options
        self.parse_options()

    def parse_options(self):
        raise NotImplementedError('Cannot instantiate base operator class')

    def _pprint_otions(self):
        return ''

    def __repr__(self):
        s = '%s (name=%s)\n' % (self.opname, self.name)
        for opt in self._supported_options:
            s += ' Option: %s=%s\n' % (opt, getattr(self, opt))
        return s


class AddOperator(Operator):
    def parse_options(self):
        pass


class AveragePool2DOperator(Operator):
    def parse_options(self):
        pass


class ResizeBilinearOperator(Operator):
    def parse_options(self):
        options = self._flatbuf_op.BuiltinOptions()
        o = ResizeBilinearOptions()
        o.Init(options.Bytes, options.Pos)
        self.align_corners = o.AlignCorners()
        self._flatbuf_options_obj = options
        self._supported_options = ['align_corners']


class Conv2DOperator(Operator):

    padding_schemes = ['SAME', 'VALID']

    def parse_options(self):
        options = self._flatbuf_op.BuiltinOptions()
        o = Conv2DOptions()
        o.Init(options.Bytes, options.Pos)
        self.stride = (o.StrideH(), o.StrideW())
        self.padding = self.padding_schemes[o.Padding()]
        self._flatbuf_options_obj = options
        self._supported_options = ['stride', 'padding']


class DepthwiseConv2DOperator(Conv2DOperator):
    def parse_options(self):
        options = self._flatbuf_op.BuiltinOptions()
        o = DepthwiseConv2DOptions()
        o.Init(options.Bytes, options.Pos)
        self.stride = (o.StrideH(), o.StrideW())
        self.padding = self.padding_schemes[o.Padding()]
        self.depth_multiplier = o.DepthMultiplier()
        self._flatbuf_options_obj = options
        self._supported_options = ['stride', 'padding', 'depth_multiplier']


operator_map = {
    'ADD': AddOperator,
    'AVERAGE_POOL_2D': AveragePool2DOperator,
    'CONV_2D': Conv2DOperator,
    'DEPTHWISE_CONV_2D': DepthwiseConv2DOperator,
    'RESIZE_BILINEAR': ResizeBilinearOperator
}


class InvalidTensorDataType(Exception):
    pass


class Tensor:

    # supported keys:
    #
    #  name: Name of this tensor
    #  shape: Shape
    #  zero_point: Quantization zero point
    #  scale: Quantization scale
    #  data_type: Date type (either INT32 or UINT8)
    #  data: Either a scalar (such as 0), None, or a numpy array of shape
    #        `shape` and entries of type `data_type`.

    # we only care about quantized models and so only care about these data
    # types.
    data_types = [None, None, 'INT32', 'UINT8']

    # It doesn't make sense to instantiate Tensor objects not in the context of
    # a specific flatbuffer model object. Basically, if this is not set to a
    # proper `model.Buffers` function, then we cannot create Tensor objects
    # (because we cannot load the data of the tensors). For testing purposes, it
    # is possible to load a tensor without loading the data, in which case this
    # function is not used.
    model_buffers = None

    def __init__(self, flatbuf_tensor, parse_data=True):
        self._flatbuf_tensor = flatbuf_tensor
        self.name = flatbuf_tensor.Name()
        self.shape = tuple(flatbuf_tensor.ShapeAsNumpy())
        self.zero_point = None
        self.scale = None
        self.data_type = None
        self.data = None
        # indicates whether or not to print the actual data stored in this
        # tensors buffer. Default to False because it's really messy.
        self.print_data = False
        self._set_quantization_params()
        if parse_data:
            self._load_data()

    def __repr__(self):
        d = self.data if self.print_data else type(self.data)
        return '%s: quant=(Z=%s, S=%s), shape=%s, data_type=%s, data=%s' % (
            self.name, self.zero_point, self.scale, self.shape,
            self.data_type, d
        )

    def _set_quantization_params(self):
        quantization = self._flatbuf_tensor.Quantization()
        zero_point = quantization.ZeroPointAsNumpy()
        if type(zero_point) == np.ndarray:
            zero_point = zero_point[0]
        self.zero_point = zero_point
        scale = quantization.ScaleAsNumpy()
        if type(scale) == np.ndarray:
            scale = scale[0]
        self.scale = scale

    def _cast(self, data, new_data_typ):
        # currently only support new_data_typ == 'INT32'
        bytelen = None
        np_typ = None
        if new_data_typ == 'INT32':
            np_typ = np.dtype('int32')
            bytelen = 4
        else:
            raise InvalidTensorDataType('Unknwon data type: %s' % (
                new_data_typ,))

        if len(data) % bytelen:
            raise ValueError('data not a multiple of type size')

        data1 = list()
        i = 0
        while i < len(data):
            x = 0
            for j in range(bytelen):
                x |= data[i + j] << (8 * j)
            i += bytelen
            data1.append(x)
        self.data_type = new_data_typ
        return np.asarray(data1, dtype=np_typ)


    def _load_data(self):
        data_idx = self._flatbuf_tensor.Buffer()
        data_typ = self.data_types[self._flatbuf_tensor.Type()]
        if data_typ is None:
            raise InvalidTensorDataType('Invalid data type: %s' % (
                self._flatbuf_tensor.Type(), ))
        data = self.model_buffers(data_idx)
        data = data.DataAsNumpy()
        self.data_type = data_typ
        if type(data) != np.ndarray:
            # probably a scalar
            self.data = data
        else:
            if data_typ != 'UINT8':
                data = self._cast(data, data_typ)
            try:
                data = data.reshape(self.shape)
            except:
                print 'Could not reshape %s to %s' % (self.name, self.shape)
            self.data = data


class TFLiteModel:

    def __init__(self, model_path, parse_data=True):
        self.model_path = model_path
        self.model = load_model(model_path)
        self.graph = self.model.Subgraphs(0)
        self.opcodes = load_opcodes(self.model)

        Operator.valid_opcodes = self.opcodes
        if parse_data:
            Tensor.model_buffers = self.model.Buffers

        self.operators = []
        self.tensors = []
        self._current_iter_idx = 0
        self._load(parse_data)

    def _load(self, parse_data):
        # load operators
        num_ops = self.graph.OperatorsLength()
        for i in range(num_ops):
            op = self.graph.Operators(i)
            op_cls = operator_map[self.opcodes[op.OpcodeIndex()]]
            self.operators.append(op_cls(op))

        # load tensors
        num_tensors = self.graph.TensorsLength()
        for i in range(num_tensors):
            fb_tensor = self.graph.Tensors(i)
            self.tensors.append(Tensor(fb_tensor, parse_data=parse_data))

    def get_input(self):
        return self.tensors[self.graph.Inputs(0)]

    def set_input(self, data, reshape=True):
        t = self.get_input()
        if reshape:
            try:
                data = data.reshape(t.shape)
                t.data = data
            except:
                print 'Could not reshape input from %s to %s' % (
                    data.shape, t.shape
                )
        else:
            t.data = data

    def get_output(self):
        return self.tensors[self.graph.Outputs(0)]

    def __iter__(self):
        return self

    def next(self):
        if self._current_iter_idx >= len(self.operators):
            raise StopIteration
        else:
            idx = self._current_iter_idx
            self._current_iter_idx += 1
            return self.operators[idx]

    def __repr__(self):
        return 'Version %s TFLite model (%s)' % (
            self.model.Version(), self.model_path)


if __name__ == '__main__':

    from sys import argv

    if len(argv) < 2:
        print 'Usage: %s [model_path]' % (argv[0],)
        exit(0)

    model = TFLiteModel(argv[1])
    for op in model:
        print '---------------------------'
        print op
        print 'Inputs:'
        for idx in op.inputs:
            t = model.tensors[idx]
            # t.print_data = True
            print '', t
        print '\nOutputs:'
        for idx in op.outputs:
            t = model.tensors[idx]
            # t.print_data = True
            print '',t

        # uncomment to stop before each layer
        # raw_input('...')
