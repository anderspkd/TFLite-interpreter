# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers

class MaximumMinimumOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsMaximumMinimumOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MaximumMinimumOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def MaximumMinimumOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # MaximumMinimumOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def MaximumMinimumOptionsStart(builder): builder.StartObject(0)
def MaximumMinimumOptionsEnd(builder): return builder.EndObject()
