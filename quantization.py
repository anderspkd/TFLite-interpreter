import numpy as np
import random
import decimal

int32 = np.int32
int64 = np.int64

INT32_MIN = int32(1 << 31)
INT32_MAX = int32(INT32_MIN - 1)

def compute_multiplier_for_conv2d(s1, s2, s3):
    # given rm = (s1*s2)/s3, compute qm, n, so that rm \approx qm >> n.
    return compute_multiplier((s1*s2)/s3)


# handlers and helpers for quantization stuff
def compute_multiplier(real_multiplier):

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


def quantized_multiplier_mult(x, multiplier, shift, exact_rounding=True, numpy=False):
    # returns (x*m)*2^{shift}
    
    # For now let's assume that the shift is a right shift
    assert shift <= 0
    
    lshift, rshift = (shift, 0) if shift > 0 else (0, -shift)
    assert lshift == 0
    if numpy:
        num = np.int64(multiplier) * np.int64(x) * (1 << lshift)
        den = np.int64(1 << (rshift + 31))
        return round(num / den)
    else:          
        # 1. saturate mult between `x * (1 << lshift)` and `multiplier`
        x_ = int32(x * (1 << lshift))
        overflow = (x_ == multiplier and x == INT32_MIN)
        a = int64(x_)
        a = a * int64(multiplier)
        nudge = int32(1 << 30) if a >= 0 else int32(1 - (1 << 30))
        a = int32((a + nudge) / int64(1 << 31))
        a = INT32_MAX if overflow else a
        
        # Just for the fun of it
        assert overflow != True
    
        # exact rounding to nearest. This is what gemmlowp implements
        # exponent = rshift
        mask = (1l << rshift) - 1
        remainder = a & mask
        # MaskIfLessThan(a, 0) & 1
        bit1 = 1 if a < 0 else 0
        threshold = (mask >> 1) + bit1
        # MaskIfGreaterThan(remainder, threshold) & 1
        bit2 = 1 if remainder > threshold else 0
    
        # v == round(a / 2^{shift})
        v = (a >> rshift) + bit2
    
        if exact_rounding:
            # in case of exact rounding, we simply return v.
            return v
    
        # otherwise, we introduce an error that is identical to the one introduced
        # by the TruncPR protocol of Caterina et al. Since we're in the clear, we
        # can cheat a bit and just use FP.
        y = float(a) / pow(2,rshift)
        alpha = np.abs(y - v)  # dist. from nearest int
        if random.random() < alpha:
            return v
        return int(y)
