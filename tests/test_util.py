import itertools

import numpy as np
import pandas as pd

from active_learning_ratio_estimation.util import outer_prod_shape_to_meshgrid_shape, dataframe_sample_statistics
from tests.util import assert_named_array_equal


def test_outer_prod_shape_to_meshgrid_shape():
    a = np.linspace(0, 10, 11)
    b = np.linspace(0, 5, 6)
    A, B = np.meshgrid(a, b)
    sum_ab = np.array(list(itertools.product(a, b))).sum(axis=1)
    sum_AB = A + B
    np.testing.assert_array_equal(outer_prod_shape_to_meshgrid_shape(sum_ab, A), sum_AB)


def test_dataframe_sample_statistics():
    n = 5
    dfs = [pd.DataFrame(np.random.rand(10, 3)) for _ in range(n)]
    expected_arrays = map(np.array, dataframe_sample_statistics(dfs))
    all_samples = np.stack([df.values for df in dfs])
    arrs = {
        'mean': all_samples.mean(axis=0),
        'std': all_samples.std(axis=0, ddof=1),
        'stderr': all_samples.std(axis=0, ddof=1)/np.sqrt(n),
    }
    for array_name, expected_arr in zip(arrs, expected_arrays):
        arr = arrs[array_name]
        assert_named_array_equal(expected_arr, arr, array_name)
