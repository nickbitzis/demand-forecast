"""Shared fixtures: the full two-year dataset (fast to generate), and one quick pipeline run per session."""

from __future__ import annotations

import pytest

from demandcast.app import json_roundtrip
from demandcast.forecast import Panels
from demandcast.generator import GenConfig, generate
from demandcast.models import GBMConfig
from demandcast.pipeline import PipelineConfig, build

SMALL_GBM = GBMConfig(max_iter=40)


@pytest.fixture(scope="session")
def ds():
    return generate(GenConfig(seed=42))


@pytest.fixture(scope="session")
def small_ds():
    return generate(GenConfig(seed=7, days=420))


@pytest.fixture(scope="session")
def small_panels(small_ds):
    return Panels.build(small_ds)


@pytest.fixture(scope="session")
def art():
    return json_roundtrip(build(PipelineConfig.quick(), log=lambda m: None))
