# import sys

# sys.path.append("/Users/richardevans/Docs/Economics/OSE/CompMethods/code/")

import pytest

# from junk_funcs import junk_func_add


def junk_func_add(arg1, arg2):
    """
    This is just a junk function that duplicates the `junk_funcs.py` module in
    the `/code/` directory. We can delete this as soon as we have some real
    functions.
    """
    junk_sum = arg1 + arg2

    return junk_sum


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [(2, 3, 5), (10, 17, 27)],
    ids=[
        "2 plus 3 equals 5",
        "10 plus 17 equals 27",
    ],
)
def test_junk_func_add(arg1, arg2, expected):
    """
    This is just a fake test of code in the `/code/` directory. We can delete
    this as soon as we have some real tests.
    """
    junk_sum = junk_func_add(arg1, arg2)

    assert junk_sum == expected
