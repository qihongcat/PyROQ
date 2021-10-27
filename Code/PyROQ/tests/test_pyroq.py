# -*- coding: utf-8 -*-
# Copyright (C) 2021 Cardiff University

"""Test suite for `PyROQ.pyroq` module
"""

__author__ = "Duncan Macleod <macleoddm@cardiff.ac.uk>"

import numpy
import numpy.testing

import pytest

from .. import pyroq


@pytest.mark.parametrize(("a", "b", "result"), (
    (numpy.array((1, 0, 0)),  # a
     numpy.array((0, 1, 0)),  # b
     numpy.array((0, 0, 0)),  # result
     ),
    (numpy.array((1, 1, 0)),
     numpy.array((0, 1, 0)),
     numpy.array((0.5, 0.5, 0)),
     ),
))
def test_proj(a, b, result):
    """Test `PyROQ.pyroq.proj` function over a few different cases
    """
    numpy.testing.assert_array_equal(
        pyroq.proj(a, b),
        result,
    )
