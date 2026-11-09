"""Shared fixtures."""

import pytest


def _stand_ins(mp):
    import fake_steps
    from harpor.webapp import jobs, steps

    mp.setattr(steps, "command", fake_steps.command)
    mp.setattr(jobs, "ask_provider", fake_steps.ask_provider)
    return fake_steps


@pytest.fixture
def fake_stages(monkeypatch):
    """Every stage of a job runs its stand-in from tests/fake_steps.py, and a well-formed API key passes without a
    request to its provider."""
    return _stand_ins(monkeypatch)


@pytest.fixture(scope="module")
def fake_stages_module():
    """fake_stages for a module-scoped fixture, for the rest of its module."""
    with pytest.MonkeyPatch.context() as mp:
        yield _stand_ins(mp)
