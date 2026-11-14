"""The id ``x`` of a streamline: ``(index << 32) | s``, int64."""

import numpy as np

from ..ids import decode, encode


def test_the_first_tractogram_has_x_equal_to_s():
    assert encode(0, 5) == 5
    np.testing.assert_array_equal(encode(0, np.arange(4)), np.arange(4))


def test_the_index_is_in_the_high_32_bits():
    assert encode(1, 5) == 4294967301  # 2 ** 32 + 5
    assert encode(3, 2 ** 32 - 1) == 4 * 2 ** 32 - 1
    assert encode(1, np.int32(7)).dtype == np.int64


def test_decode_inverts_encode_vectorised():
    index = np.array([0, 0, 1, 2, 70000])
    s = np.array([0, 2 ** 31 - 1, 0, 12, 2 ** 32 - 1], dtype=np.int64)
    x = encode(index, s.astype(np.int64))
    assert x.dtype == np.int64
    got_index, got_s = decode(x)
    np.testing.assert_array_equal(got_index, index)
    np.testing.assert_array_equal(got_s, s)
    assert decode(4294967301) == (1, 5)
