# SPDX-License-Identifier: MIT
"""Test bootstrap: import moshi.sim_* without the torch-heavy package __init__.

moshi/__init__.py imports models and modules, which import torch. The Sim session layer needs
neither torch nor a GPU, so tests register a bare 'moshi' package that points at the source
directory. If the real package is already imported (torch installed), it is used as-is.
"""

import os
import sys
import types

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))  # holds the moshi package
PACKAGE_DIR = os.path.join(PROJECT_DIR, "moshi")
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, os.pardir))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

if "moshi" not in sys.modules:
    _stub = types.ModuleType("moshi")
    _stub.__path__ = [PACKAGE_DIR]
    _stub.__file__ = os.path.join(PACKAGE_DIR, "__init__.py")
    sys.modules["moshi"] = _stub
