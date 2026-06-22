"""Fixtures for the tutorial-notebook smoke tests.

Each tutorial test executes one notebook end-to-end in a fresh kernel and fails
if any cell raises.  The notebooks themselves are never modified: they are read
into memory, executed there, and the executed copy is written to a throwaway
temp file purely for post-mortem inspection.

The ``tutorial`` marker that gates these tests is registered in
``pyproject.toml``; exclude them with ``pytest -m "not tutorial"``.
"""

from pathlib import Path

import pytest

TUTORIALS_DIR = Path(__file__).resolve().parents[2] / "tutorials"

# Per-cell execution timeout (seconds).  Some tutorials run multi-second
# simulations in a single cell, so this is intentionally generous.
CELL_TIMEOUT = 1800


def _run_notebook(notebook_name, log_path):
    """Execute ``tutorials/<notebook_name>`` end-to-end, raising on any error.

    The notebook is run with its working directory set to ``tutorials/`` so that
    its relative paths (``../src/...``, ``demo_output/...``) resolve exactly as
    they do for a user running the notebook interactively.  The original file on
    disk is left untouched; the executed copy is written to ``log_path``.
    """
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    notebook_path = TUTORIALS_DIR / notebook_name
    nb = nbformat.read(notebook_path, as_version=4)
    ep = ExecutePreprocessor(timeout=CELL_TIMEOUT, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(TUTORIALS_DIR)}})
    finally:
        # Persist whatever ran (including the cell that failed) for debugging.
        nbformat.write(nb, str(log_path))


@pytest.fixture
def notebook_runner(tmp_path):
    """Return a callable ``run(notebook_name)`` that executes a tutorial notebook.

    Cell outputs are streamed into a temporary copy of the executed notebook
    under pytest's ``tmp_path`` so they never pollute the repository.
    """

    def run(notebook_name):
        log_path = tmp_path / notebook_name
        _run_notebook(notebook_name, log_path)

    return run
