import builtins
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

# PySide6 replaces builtins.__import__ with a hook that inspects the source of
# every imported module (shibokensupport.feature._mod_uses_pyside). In a
# combined test run, torch.package has also patched inspect.getfile, so when
# matplotlib/dateutil later does `from six.moves import ...`, the hook fails
# with `_SixMetaPathImporter has no attribute '_path'`.
#
# The test suite does not use PySide6's `__feature__` imports, so restore the
# original import function instead of letting the hook run for every import.
if hasattr(builtins, "__orig_import__"):
    builtins.__import__ = builtins.__orig_import__


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
