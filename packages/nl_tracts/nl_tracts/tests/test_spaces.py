import warnings

import pytest

from nl_tracts.spaces import SpaceMismatchError, check_same_space


def test_equal_spaces_pass():
    check_same_space("MNI152NLin2009cAsym", "MNI152NLin2009cAsym")


@pytest.mark.parametrize(
    "a,b",
    [
        ("MNI152NLin2009cAsym", "MNI152NLin6Asym"),
        ("subject:100307", "subject:100408"),
        ("MNI152NLin2009cAsym", "subject:100307"),
        ("mni", "MNI"),  # tags are compared as they are
    ],
)
def test_different_spaces_raise(a, b):
    with pytest.raises(SpaceMismatchError, match=a):
        check_same_space(a, b)


def test_mismatch_is_a_value_error():
    assert issubclass(SpaceMismatchError, ValueError)


def test_override_warns_instead_of_raising():
    with pytest.warns(UserWarning, match="subject:1.*subject:2"):
        check_same_space("subject:1", "subject:2", allow_mismatch=True)


def test_override_is_silent_when_the_spaces_agree():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_same_space("x", "x", allow_mismatch=True)
