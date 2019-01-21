import numpy as np
import Decimals

int32 = np.int32
int64 = np.int64


# handlers and helpers for quantization stuff
def compute_multiplier(s1, s2, s3):
    # given three FP32 values (corresponding to three quantization scale
    # values), compute and return integers n and qm, so that
    #
    #   (s1*s2)/s3 = qm/2^n
    #
    # qm fits in a int32

    real_multiplier = (s1*s2)/s3

    assert 0.0 < real_multiplier < 1.0, \
        'real_multiplier=%s (s1=%s, s2=%s, s3=%s)' % (
            real_multiplier, s1, s2, s3)

    n = 0
    while (real_multiplier < 0.5):
        real_multiplier *= 2.0;
        n += 1

    sh = 1 << 31
    qm = int64(np.round(real_multiplier * sh))

    if qm == sh:
        qm /= 2
        n -= 1

    return n, int32(qm)

INT32_MIN = int32(1 << 31)
INT32_MAX = int32(INT32_MIN - 1)

# TODO: Figure out a way to test this guy
def quantized_multiplier_mult(x, multiplier, shift):
    lshift, rshift = (shift, 0) if shift > 0 else (0, -shift)

    # 1. saturate mult between `x * (1 << lshift)` and `multiplier`
    x *= 1 << lshift
    overflow = (x == multiplier and x == INT32_MIN)
    a = int64(x)
    a = a * int64(multiplier)
    nudge = (1 << 30) if a >= 0 else (1 - (1 << 30))
    a = int32((a + nudge)/int64(1 << 31))
    a = INT32_MAX if overflow else a

    # 2. rounding to nearest
    mask = (1 << shift) - 1
    remainder = a & mask
    bit1 = 1 if a < 0 else 0
    threshold = (mask >> 1) + bit1
    bit2 = 1 if remainder > threshold else 0
    return (a << shift) + bit2
