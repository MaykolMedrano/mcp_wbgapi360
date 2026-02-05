
import pytest
import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



@pytest.fixture
def api_client():
    from wbgapi360.core.client import Data360Client
    return Data360Client()
