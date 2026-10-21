"""Shared test fixtures: wire payloads, IMU samples and synthetic camera scenes.

Import from test modules as `from support.wire import frame`; unittest
discovery (`-s tests`) puts the tests directory on sys.path.
"""

import importlib
import unittest
from pathlib import Path
from types import ModuleType

# Repository root, for tests that load project scripts by path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def require(*module_names: str) -> tuple[ModuleType, ...]:
    """Import optional dependencies, or skip the calling test if one is missing.

    Example: `cv2, np = require("cv2", "numpy")` in setUp or setUpClass.
    """
    modules = []
    for name in module_names:
        try:
            modules.append(importlib.import_module(name))
        except ImportError:
            raise unittest.SkipTest(f"Optional dependency {name} unavailable")
    return tuple(modules)
