# -*- coding: utf-8 -*-

"""
keras_resnet.blocks
~~~~~~~~~~~~~~~~~~~

This module implements a number of popular residual blocks.
"""

from ._1d import (
    Basic1D,
    Bottleneck1D
)

from ._2d import (
    basic_2d,
    bottleneck_2d
)

from ._3d import (
    basic_3d,
    bottleneck_3d
)

from ._time_distributed_2d import (
    time_distributed_basic_2d,
    time_distributed_bottleneck_2d
)
