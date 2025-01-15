import numpy as np
import pytest
from flygym.examples.olfaction import OdorPlumeArena
from flygym.util import get_data_path


def test_plume_arena():
    plume_data_path = get_data_path(
        "flygym", "data/test_data/plume_tracking/plume_short.hdf5"
    )
    arena = OdorPlumeArena(
        plume_data_path=plume_data_path,
        main_camera_name="",  # empty string as no camera is added to the arena for this test
        plume_simulation_fps=20,
        intensity_scale_factor=1.0,
    )
    positions_1 = np.array([[0, 80, 1], [10, 80, 1], [52, 80, 1], [75, 105, 1]])
    positions_2 = np.array([[75, 15, 1], [10, 80, 1], [239, 159, 1], [240, 160, 1]])
    intensities_1 = arena.get_olfaction(positions_1)
    intensities_2 = arena.get_olfaction(positions_2)
    assert intensities_1[0, :] == pytest.approx([0, 0.45141602, 0.58837891, 0.17651369])
    assert intensities_2[0, :3] == pytest.approx([0, 0.45141602, 0])
    assert np.isnan(intensities_2[0, 3])  # out of bound
