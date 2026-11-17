import os
from pathlib import Path
import tempfile

_test_dir = tempfile.TemporaryDirectory(prefix='researchcompress-tests-')
os.environ['STORAGE_PATH'] = _test_dir.name
os.environ['DATABASE_URL'] = 'sqlite:///' + (Path(_test_dir.name) / 'test.db').as_posix()
os.environ['ENABLE_BASELINE_COMPARISON'] = 'true'

import pytest
import pymupdf
from fastapi.testclient import TestClient
from app.main import app
from app.db import Base, engine


def pytest_sessionfinish(session, exitstatus):
    engine.dispose()


@pytest.fixture
def client():
    Base.metadata.drop_all(engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def pdf_bytes():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((60, 70), 'A Controlled Scientific Study', fontsize=18)
    page.insert_text((60, 110), 'Abstract\nWe study accurate compression of scientific text. We evaluate a small model on a fixed dataset.', fontsize=11)
    page.insert_text((60, 180), 'Methods\nWe used a controlled experiment with 40 samples and a held-out evaluation dataset.', fontsize=11)
    page.insert_text((60, 250), 'Results\nThe method achieved 92.5% accuracy. The control achieved 85.0% accuracy.', fontsize=11)
    page.insert_text((60, 320), 'Limitations\nThe study used only one dataset. These results do not establish generalization.', fontsize=11)
    doc.set_metadata({'title': 'A Controlled Scientific Study', 'author': 'Test Author'})
    result = doc.tobytes()
    doc.close()
    return result


class TestTokenizer:
    """Character tokenizer for deterministic boundary tests, never used in runtime."""
    def encode(self, text, add_special_tokens=False):
        return list(text) + (['BOS', 'EOS'] if add_special_tokens else [])
    def num_special_tokens_to_add(self, pair=False):
        return 2
    def __call__(self, text, **kwargs):
        return {'offset_mapping': [(i, i+1) for i in range(len(text))]}


@pytest.fixture
def tokenizer():
    return TestTokenizer()
