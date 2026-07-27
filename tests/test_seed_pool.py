import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from extract_embeddings import get_embedding_and_attributes_robust


class Face:
    bbox = np.array([0, 0, 20, 30], dtype=np.float32)
    normed_embedding = np.ones(512, dtype=np.float32)
    age = 30
    gender = 1


class App:
    def __init__(self):
        self.calls = []
    def prepare(self, ctx_id, det_size):
        self.calls.append((ctx_id, det_size))
    def get(self, image):
        return [Face()] if len(self.calls) == 2 else []


def test_raw_seed_embedding_retries_larger_detectors():
    app = App()
    embedding, age, gender = get_embedding_and_attributes_robust(
        app, np.zeros((100, 100, 3), dtype=np.uint8), 0
    )
    assert embedding.shape == (512,)
    assert (age, gender) == (30, 1)
    assert app.calls[:2] == [(0, (640, 640)), (0, (512, 512))]


def test_raw_seed_embedding_handles_unreadable_image():
    assert get_embedding_and_attributes_robust(App(), None, 0) == (None, None, None)
