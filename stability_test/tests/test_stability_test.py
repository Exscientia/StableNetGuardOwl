"""
Unit and regression test for the stability_test package.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest
import stability_test


def test_stability_test_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "stability_test" in sys.modules
