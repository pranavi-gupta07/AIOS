import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    os.environ["AIOS_MODEL_DIR"] = str(d)
    return d


@pytest.fixture(scope="session")
def store(model_dir):
    from aios.models import ModelStore
    s = ModelStore(model_dir)
    s.ensure_zoo()
    return s
