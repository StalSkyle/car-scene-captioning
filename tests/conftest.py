import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_source_module(module_name: str, relative_path: str):
    """Load a project module directly from its file without executing autocaption/__init__.py."""
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {relative_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def image_loader_module():
    return load_source_module(
        "_image_loader_under_test",
        "autocaption/image_loader.py",
    )


@pytest.fixture(scope="session")
def image_processor_module():
    return load_source_module(
        "_image_processor_under_test",
        "autocaption/image_processor.py",
    )
