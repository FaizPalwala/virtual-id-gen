"""Tests for ArcFace embedding extraction utilities."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from extract_embeddings import (
    age_to_group,
    get_embedding_and_attributes,
)


class TestAgeToGroup:
    def test_negative_age_is_unknown(self):
        assert age_to_group(-1) == -1
        assert age_to_group(-999) == -1

    def test_young_boundary(self):
        assert age_to_group(0) == 0
        assert age_to_group(24) == 0

    def test_adult_boundary(self):
        assert age_to_group(25) == 1
        assert age_to_group(44) == 1

    def test_middle_aged_boundary(self):
        assert age_to_group(45) == 2
        assert age_to_group(64) == 2

    def test_senior(self):
        assert age_to_group(65) == 3
        assert age_to_group(120) == 3

    def test_age_groups_cover_all_four_categories(self):
        # The four stable proxy classes plus the unknown (-1) sentinel.
        assert set(age_to_group(a) for a in (-1, 0, 24, 25, 44, 45, 64, 65, 120)) == {0, 1, 2, 3, -1}


class Face:
    def __init__(self, embedding, det_score=0.95, age=30, gender=1):
        self.normed_embedding = embedding
        self.det_score = det_score
        self.age = age
        self.gender = gender


class AppWithFaces:
    def __init__(self, faces):
        self._faces = faces

    def get(self, _image):
        return self._faces


class TestGetEmbeddingAndAttributes:
    def test_returns_largest_face_by_det_score(self):
        low = Face(np.arange(512, dtype=np.float32), det_score=0.6)
        high = Face(np.ones(512, dtype=np.float32), det_score=0.99)
        app = AppWithFaces([low, high])
        emb, age, gender = get_embedding_and_attributes(app, np.zeros((100, 100, 3), dtype=np.uint8))
        assert np.array_equal(emb, np.ones(512, dtype=np.float32))
        assert (age, gender) == (30, 1)

    def test_no_faces_returns_none(self):
        app = AppWithFaces([])
        emb, age, gender = get_embedding_and_attributes(app, np.zeros((10, 10, 3), dtype=np.uint8))
        assert emb is None
        assert age is None
        assert gender is None

    def test_missing_age_gender_defaults_to_minus_one(self):
        f = Face(np.zeros(512, dtype=np.float32))
        del f.age
        del f.gender
        app = AppWithFaces([f])
        emb, age, gender = get_embedding_and_attributes(app, np.zeros((10, 10, 3), dtype=np.uint8))
        assert emb is not None
        assert age == -1
        assert gender == -1
