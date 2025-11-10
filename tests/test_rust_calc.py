import pytest
import numpy as np
from gloshbridge import GLOSHBridge


@pytest.fixture
def sample_test_data_path():
    return "tests/datasets/toy/toy.csv"


def test_glosh_scores_rust_1(sample_test_data_path):
    glosh_scores = GLOSHBridge(sample_test_data_path, min_pts=5, min_clsize=3).calc_rust_outlier_scores()

    expected = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.10557281,
            0.0,
            0.36754447,
            0.33333333,
            0.292893218,
            0.0,
            0.4452998,
            0.2094305827,
            0.0,
            0.292893218,
            0.0,
            0.292893218,
            0.292893218,
            0.51492875,
        ]
    )

    np.testing.assert_allclose(glosh_scores, expected, rtol=1e-6)


def test_glosh_scores_rust_2(sample_test_data_path):
    glosh_scores = GLOSHBridge(sample_test_data_path, min_pts=3, min_clsize=3).calc_rust_outlier_scores()

    expected = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.64644661, 0.75])

    np.testing.assert_allclose(glosh_scores, expected, rtol=1e-6)


def test_glosh_scores_rust_3(sample_test_data_path):
    glosh_scores = GLOSHBridge(sample_test_data_path, min_pts=5, min_clsize=7).calc_rust_outlier_scores()

    expected = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.105573,
            0.0,
            0.367544,
            0.333333,
            0.0,
            0.0,
            0.4453,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.514929,
        ]
    )

    np.testing.assert_allclose(glosh_scores, expected, rtol=1e-5)
