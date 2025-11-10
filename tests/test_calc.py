import pytest
import numpy as np
from gloshbridge.glosh_calc_cor import GLOSH


@pytest.fixture
def sample_data():
    """Generate a small synthetic dataset for testing."""
    data = [
        [2, 9],
        [3, 9],
        [4, 9],
        [2, 8],
        [3, 8],
        [4, 8],
        [1, 6],
        [2, 6],
        [1, 5],
        [2, 5],
        [5, 2],
        [6, 2],
        [5, 1],
        [6, 1],
        [8, 4],
        [9, 4],
        [8, 3],
        [9, 3],
        [6, 5],
        [8, 8],
    ]
    return np.array(data)


@pytest.fixture
def glosh_instance(sample_data):
    """Create a GLOSH instance."""
    return GLOSH(sample_data, min_pts=3, min_clsize=3)


def test_glosh_scores(glosh_instance):
    """Ensure GLOSH scores are computed correctly."""
    glosh_scores = glosh_instance.calc_glosh_scores()

    expected = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.64644661, 0.75])

    np.testing.assert_allclose(glosh_scores, expected, rtol=1e-6)


def test_glosh_scores_2(sample_data):
    """Ensure GLOSH scores are computed correctly."""
    glosh_instance_alt = GLOSH(sample_data, min_pts=5, min_clsize=3)
    glosh_scores = glosh_instance_alt.calc_glosh_scores()

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


def test_glosh_scores_3(sample_data):
    glosh_instance_alt = GLOSH(sample_data, min_pts=5, min_clsize=7)
    glosh_scores = glosh_instance_alt.calc_glosh_scores()

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
