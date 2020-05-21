import itertools

import numpy as np

from active_learning_ratio_estimation.util import outer_prod_shape_to_meshgrid_shape


def test_outer_prod_shape_to_meshgrid_shape():
    a = np.linspace(0, 10, 11)
    b = np.linspace(0, 5, 6)
    A, B = np.meshgrid(a, b)
    sum_ab = np.array(list(itertools.product(a, b))).sum(axis=1)
    sum_AB = A + B
    np.testing.assert_array_equal(outer_prod_shape_to_meshgrid_shape(sum_ab, A), sum_AB)
