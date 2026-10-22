"""Shared fixtures for the web-prototype tests (snapshot, notify, API service)."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from mcxrv import pipeline, snapshot


@pytest.fixture(scope="session")
def small_result():
    """One short synthetic run shared by every test that needs real pipeline output."""
    cfg = pipeline.PipelineConfig(source="synthetic", start=date(2025, 6, 1),
                                  end=date(2026, 9, 18), seed=7)
    return pipeline.run(cfg)


@pytest.fixture(scope="session")
def small_snapshot(small_result):
    return snapshot.build(small_result,
                          generated_at=datetime(2026, 9, 18, 18, 0, tzinfo=timezone.utc))
