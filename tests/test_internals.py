"""Tests for internal helpers in pdex.__init__."""

import numpy as np
import pandas as pd
import pytest

from pdex import _identify_reference_index, _unique_groups, _validate_groupby


class TestValidateGroupby:
    def test_valid_column(self):
        obs = pd.DataFrame({"guide": ["A", "B"], "batch": [1, 2]})
        _validate_groupby(obs, "guide")  # should not raise

    def test_missing_column(self):
        obs = pd.DataFrame({"guide": ["A", "B"]})
        with pytest.raises(ValueError, match="Missing column.*nonexistent"):
            _validate_groupby(obs, "nonexistent")

    def test_error_lists_available_columns(self):
        obs = pd.DataFrame({"guide": ["A"], "batch": [1]})
        with pytest.raises(ValueError, match="guide"):
            _validate_groupby(obs, "missing")


class TestIdentifyReferenceIndex:
    def test_found(self):
        groups = np.array(["A", "non-targeting", "B"])
        assert _identify_reference_index(groups, "non-targeting") == 1

    def test_missing(self):
        groups = np.array(["A", "B", "C"])
        with pytest.raises(ValueError, match="Missing reference"):
            _identify_reference_index(groups, "non-targeting")

    def test_duplicate(self):
        groups = np.array(["A", "non-targeting", "non-targeting"])
        with pytest.raises(ValueError, match="Multiple references"):
            _identify_reference_index(groups, "non-targeting")


class TestUniqueGroups:
    def test_basic(self):
        obs = pd.DataFrame({"guide": ["A", "B", "A", "B", "C"]})
        groups, indices = _unique_groups(obs, "guide")
        assert set(groups) == {"A", "B", "C"}
        assert len(indices) == 5

    def test_filters_empty_string(self):
        obs = pd.DataFrame({"guide": ["A", "", "B", ""]})
        groups, indices = _unique_groups(obs, "guide")
        assert "" not in groups
        # Filtered cells get -1 sentinel
        assert len(indices) == 4
        assert (indices[[1, 3]] == -1).all()

    def test_filters_nan(self):
        obs = pd.DataFrame({"guide": ["A", None, "B", None]})
        groups, indices = _unique_groups(obs, "guide")
        assert len(groups) == 2
        assert set(groups) == {"A", "B"}
        # Length preserved, NaN cells get -1
        assert len(indices) == 4
        assert (indices[[1, 3]] == -1).all()
