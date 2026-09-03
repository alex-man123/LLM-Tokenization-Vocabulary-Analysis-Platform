"""Unit tests for `ui/embeddings_3d.py` (Task 8.12): illustrative embeddings + PCA.

`ui/` is not on `pythonpath` (only `src/` is, `pyproject.toml`), so this
test inserts it into `sys.path` itself, exactly like every page under
`ui/pages/` already does to import sibling modules such as
`tokenizer_options`.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))

from embeddings_3d import (  # noqa: E402
    DEFAULT_EMBEDDING_DIMENSIONS,
    generate_illustrative_embeddings,
    pca_3d,
)


def test_generate_illustrative_embeddings_shape():
    vectors = generate_illustrative_embeddings([1, 2, 3])

    assert vectors.shape == (3, DEFAULT_EMBEDDING_DIMENSIONS)


def test_generate_illustrative_embeddings_respects_custom_dimensions():
    vectors = generate_illustrative_embeddings([1, 2], dimensions=8)

    assert vectors.shape == (2, 8)


def test_generate_illustrative_embeddings_is_deterministic_per_token_id():
    first = generate_illustrative_embeddings([42])
    second = generate_illustrative_embeddings([42])

    assert np.array_equal(first, second)


def test_generate_illustrative_embeddings_differs_across_token_ids():
    vectors = generate_illustrative_embeddings([1, 2])

    assert not np.array_equal(vectors[0], vectors[1])


def test_generate_illustrative_embeddings_of_empty_list_has_zero_rows():
    vectors = generate_illustrative_embeddings([])

    assert vectors.shape == (0, DEFAULT_EMBEDDING_DIMENSIONS)


def test_pca_3d_output_shape_matches_input_row_count():
    vectors = generate_illustrative_embeddings([1, 2, 3, 4, 5])

    coords = pca_3d(vectors)

    assert coords.shape == (5, 3)


def test_pca_3d_is_deterministic():
    vectors = generate_illustrative_embeddings([1, 2, 3])

    first = pca_3d(vectors)
    second = pca_3d(vectors)

    assert np.allclose(first, second)


def test_pca_3d_of_a_single_row_is_the_zero_vector():
    vectors = generate_illustrative_embeddings([1])

    coords = pca_3d(vectors)

    assert coords.shape == (1, 3)
    assert np.allclose(coords, 0.0)


def test_pca_3d_of_zero_rows_returns_zero_rows():
    coords = pca_3d(np.zeros((0, DEFAULT_EMBEDDING_DIMENSIONS)))

    assert coords.shape == (0, 3)


def test_pca_3d_handles_fewer_than_three_input_dimensions_without_raising():
    vectors = generate_illustrative_embeddings([1, 2, 3], dimensions=2)

    coords = pca_3d(vectors)

    assert coords.shape == (3, 3)
    # the padded (nonexistent) third principal component is always zero
    assert np.allclose(coords[:, 2], 0.0)
