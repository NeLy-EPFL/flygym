"""Tutorial tests: execute every tutorial notebook end-to-end, failing on any error.

One test is parametrized per notebook in ``tutorials/``.  Each runs the notebook
in a fresh kernel (see the ``notebook_runner`` fixture in ``conftest.py``)
without modifying the file on disk.

These tests are slow (each spins up a full simulation), so they carry the
``tutorial`` marker and can be skipped in one shot with ``pytest -m "not
tutorial"``.

Tutorials 3 and 8 additionally exercise the GPU-accelerated (warp) backend, so they
are also tagged ``warp`` and are excluded by ``-m "not warp"`` / when warp is absent.
"""

import os

import pytest

# Every tutorial spins up a full simulation that renders, so the whole module
# is skipped on runners without headless GL (which set SKIP_RENDERING_TESTS=1).
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_RENDERING_TESTS") == "1",
    reason="SKIP_RENDERING_TESTS=1 (eg. headless GL unavailable on this CI runner)",
)

NOTEBOOKS = [
    "1a_basic_model_composition.ipynb",
    "1b_advanced_model_composition.ipynb",
    "2_replaying_experimental_recordings.ipynb",
    pytest.param("3_gpu_accelerated_simulation.ipynb", marks=pytest.mark.warp),
    "4a_cpg_controller.ipynb",
    "4b_rule_based_controller.ipynb",
    "4c_hybrid_controller.ipynb",
    "4d_turning_controller.ipynb",
    "5a_replaying_experimental_flybody_onball.ipynb",
    "5b_using_flybody_model.ipynb",
    pytest.param(
        "8_recording_and_rendering_trajectories.ipynb", marks=pytest.mark.warp
    ),
]


@pytest.mark.tutorial
@pytest.mark.parametrize("notebook_name", NOTEBOOKS)
def test_tutorial_notebook(notebook_name, notebook_runner):
    notebook_runner(notebook_name)
