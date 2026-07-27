"""Tests for shared image, quality, and manifest utilities in common.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from common import (
    get_image_paths,
    laplacian_variance,
    normalised_cosine_similarity,
    VALID_IMAGE_EXTENSIONS,
)


class TestGetImagePaths:
    def test_returns_sorted_paths(self, tmp_path):
        (tmp_path / "b.jpg").touch()
        (tmp_path / "a.png").touch()
        (tmp_path / "not_an_image.txt").touch()
        result = get_image_paths(str(tmp_path))
        assert [p.name for p in result] == ["a.png", "b.jpg"]

    def test_non_recursive_excludes_subdirectories(self, tmp_path):
        (tmp_path / "top.jpg").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.png").touch()
        result = get_image_paths(str(tmp_path), recursive=False)
        assert [p.name for p in result] == ["top.jpg"]

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert get_image_paths(str(tmp_path)) == []

    def test_respects_valid_extensions_only(self, tmp_path):
        (tmp_path / "valid.jpg").touch()
        (tmp_path / "invalid.gif").touch()
        (tmp_path / "no_ext").touch()
        result = get_image_paths(str(tmp_path))
        assert [p.name for p in result] == ["valid.jpg"]

    def test_webp_included_by_default(self, tmp_path):
        (tmp_path / "image.webp").touch()
        result = get_image_paths(str(tmp_path))
        assert len(result) == 1
        assert result[0].name == "image.webp"


class TestLaplacianVariance:
    def test_blur_lower_than_sharp(self):
        rng = np.random.RandomState(42)
        sharp = (rng.rand(128, 128, 3) * 255).astype(np.uint8)
        # Box blur via downscale-upscale
        import cv2
        tiny = cv2.resize(sharp, (16, 16))
        blur = cv2.resize(tiny, (128, 128))
        assert laplacian_variance(sharp) > laplacian_variance(blur)

    def test_flat_image_is_zero(self):
        flat = np.full((64, 64, 3), 128, dtype=np.uint8)
        assert laplacian_variance(flat) == 0.0

    def test_random_noise_is_positive(self):
        rng = np.random.RandomState(7)
        noise = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
        assert laplacian_variance(noise) > 0.0


class TestNormalisedCosineSimilarity:
    def test_identical_vectors(self):
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert normalised_cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert normalised_cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        assert normalised_cosine_similarity(a, -a) == pytest.approx(-1.0)

    def test_arbitrary_angle(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = normalised_cosine_similarity(a, b)
        assert 0.0 < result < 1.0

    def test_zero_norm_raises(self):
        with pytest.raises(ValueError, match="zero-norm"):
            normalised_cosine_similarity(
                np.zeros(3, dtype=np.float32),
                np.ones(3, dtype=np.float32),
            )
