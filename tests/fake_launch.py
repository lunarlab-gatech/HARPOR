"""Run a HARPOR module with the stand-in stages, for a test that needs the server or the terminal command as a
process of its own (the fake_stages fixture patches only the test's own process):

    python tests/fake_launch.py MODULE [ARGS]     such as harpor.webapp.run cup.mp4 --runs DIR

The module name stays on the command line, where jobs.holder_alive looks for "harpor.webapp".
"""

import runpy
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))  # the repo, after this folder (fake_steps)

import fake_steps  # noqa: E402
from harpor.webapp import jobs, steps  # noqa: E402

steps.command = fake_steps.command
jobs.ask_provider = fake_steps.ask_provider
module = sys.argv[1]
sys.argv = [module, *sys.argv[2:]]
runpy.run_module(module, run_name="__main__", alter_sys=True)
