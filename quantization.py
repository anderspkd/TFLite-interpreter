import numpy as np
import decimal

int32 = np.int32
int64 = np.int64

INT32_MIN = int32(1 << 31)
INT32_MAX = int32(INT32_MIN - 1)


# handlers and helpers for quantization stuff
def compute_multiplier(s1, s2, s3):
    # given rm = (s1*s2)/s3, compute qm, n, so that rm \approx qm >> n.

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

    assert qm <= INT32_MAX
    assert n >= 0

    return -n, int32(qm)


def quantized_multiplier_mult(x, multiplier, shift):
    lshift, rshift = (shift, 0) if shift > 0 else (0, -shift)
    # 1. saturate mult between `x * (1 << lshift)` and `multiplier`
    x_ = int32(x * (1 << lshift))
    overflow = (x_ == multiplier and x == INT32_MIN)
    a = int64(x_)
    a = a * int64(multiplier)
    nudge = (1 << 30) if a >= 0 else (1 - (1 << 30))
    a = int32((a + nudge) / (1 << 31))
    a = INT32_MAX if overflow else a

    # 2. rounding to nearest
    # exponent = rshift
    mask = (1l << rshift) - 1
    remainder = a & mask
    # MaskIfLessThan(a, 0) & 1
    bit1 = 1 if a < 0 else 0
    threshold = (mask >> 1) + bit1
    # MaskIfGreaterThan(remainder, threshold) & 1
    bit2 = 1 if remainder > threshold else 0
    return (a >> rshift) + bit2
