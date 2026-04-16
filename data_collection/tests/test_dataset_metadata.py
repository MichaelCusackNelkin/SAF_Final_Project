"""
test_dataset_metadata.py

Unit tests for ``check_start_date()`` and ``DATASET_START_DATES`` in
``data_collection.dataset_metadata``.

All tests are isolated, deterministic, and require no network access.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_collection.dataset_metadata import (
    DATASET_START_DATES,
    check_start_date,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

KNOWN_DATASET = "EQUS.MINI"
# The canonical start date for EQUS.MINI as declared in production code.
DATASET_START = pd.Timestamp(DATASET_START_DATES[KNOWN_DATASET])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ts(date_str: str) -> pd.Timestamp:
    """
    Return a timezone-naive ``pd.Timestamp`` from an ISO-8601 date string.

    Parameters
    ----------
    date_str : str
        Date in ``"YYYY-MM-DD"`` format.

    Returns
    -------
    pd.Timestamp
    """
    return pd.Timestamp(date_str)


# ---------------------------------------------------------------------------
# 1. No error — start_dt is after the dataset start
# ---------------------------------------------------------------------------

class TestNoError:
    """start_dt falls after the known dataset start; must not raise."""

    def test_well_after_dataset_start(self) -> None:
        start = DATASET_START + pd.Timedelta(days=100)
        check_start_date(start, KNOWN_DATASET)  # should not raise

    def test_one_day_after_dataset_start(self) -> None:
        start = DATASET_START + pd.Timedelta(days=1)
        check_start_date(start, KNOWN_DATASET)  # should not raise


# ---------------------------------------------------------------------------
# 2. ValueError raised — start_dt is before the dataset start
# ---------------------------------------------------------------------------

class TestRaisesBeforeDatasetStart:
    """start_dt falls before the known dataset start; must raise ValueError."""

    def test_start_before_dataset_start(self) -> None:
        start = DATASET_START - pd.Timedelta(days=60)
        with pytest.raises(ValueError):
            check_start_date(start, KNOWN_DATASET)

    def test_start_far_before_dataset_start(self) -> None:
        start = _ts("2020-01-01")
        with pytest.raises(ValueError):
            check_start_date(start, KNOWN_DATASET)

    def test_start_one_day_before_dataset_start(self) -> None:
        start = DATASET_START - pd.Timedelta(days=1)
        with pytest.raises(ValueError):
            check_start_date(start, KNOWN_DATASET)

    def test_error_message_contains_dataset_name(self) -> None:
        start = DATASET_START - pd.Timedelta(days=10)
        with pytest.raises(ValueError, match=KNOWN_DATASET):
            check_start_date(start, KNOWN_DATASET)

    def test_error_message_contains_context(self) -> None:
        ctx = "clustering_pipeline"
        start = DATASET_START - pd.Timedelta(days=10)
        with pytest.raises(ValueError, match=ctx):
            check_start_date(start, KNOWN_DATASET, context=ctx)

    def test_error_message_contains_requested_date(self) -> None:
        start = _ts("2022-06-15")
        with pytest.raises(ValueError, match="2022-06-15"):
            check_start_date(start, KNOWN_DATASET)


# ---------------------------------------------------------------------------
# 3. Unknown dataset — no error
# ---------------------------------------------------------------------------

class TestUnknownDataset:
    """For an unrecognised dataset code, check_start_date must not raise."""

    def test_unknown_dataset_does_not_raise(self) -> None:
        start = _ts("2015-01-01")
        check_start_date(start, "UNKNOWN.DATASET")  # should not raise

    def test_unknown_dataset_with_very_early_start(self) -> None:
        start = _ts("2000-01-01")
        check_start_date(start, "DOES.NOT.EXIST")  # should not raise


# ---------------------------------------------------------------------------
# 4. Boundary case — start_dt equals the dataset start exactly
# ---------------------------------------------------------------------------

class TestBoundaryCase:
    """start_dt on exactly the dataset start date must not raise."""

    def test_start_equals_dataset_start(self) -> None:
        start = DATASET_START
        check_start_date(start, KNOWN_DATASET)  # should not raise


# ---------------------------------------------------------------------------
# 5. DATASET_START_DATES sanity checks
# ---------------------------------------------------------------------------

class TestDatasetStartDatesDict:
    """Smoke-tests on the metadata dict itself."""

    def test_equs_mini_present(self) -> None:
        assert "EQUS.MINI" in DATASET_START_DATES

    def test_all_values_parseable_as_timestamps(self) -> None:
        for dataset, date_str in DATASET_START_DATES.items():
            ts = pd.Timestamp(date_str)
            assert isinstance(ts, pd.Timestamp), (
                f"Could not parse '{date_str}' for dataset '{dataset}'"
            )

    def test_all_start_dates_are_in_the_past(self) -> None:
        today = pd.Timestamp.today().normalize()
        for dataset, date_str in DATASET_START_DATES.items():
            ts = pd.Timestamp(date_str)
            assert ts <= today, (
                f"Dataset '{dataset}' has a future start date: {date_str}"
            )
