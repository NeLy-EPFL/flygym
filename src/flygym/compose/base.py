from abc import ABC, abstractmethod
from pathlib import Path
from os import PathLike
from copy import deepcopy

import mujoco
import dm_control.mjcf as mjcf


class BaseCompositionElement(ABC):
    """Base class for composable elements in the MuJoCo model, providing common
    functionality such as compiling to MuJoCo model/data and exporting."""

    @property
    @abstractmethod
    def mjcf_root(self) -> mjcf.RootElement:
        pass

    def compile(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Compile the MJCF model into MuJoCo MjModel and MjData objects. This is where
        `dm_control.mjcf` "hands off" the model that it composes to `mujoco` for
        simulation. Things like the ordering of generalized coordinates (qpos) are
        determined here."""
        physics = mjcf.Physics.from_mjcf_model(self.mjcf_root)
        return physics.model._model, physics.data._data

    def save_xml_with_assets(
        self, output_dir: PathLike, xml_filename: str = None
    ) -> None:
        """Export the MJCF model to a directory, including a XML file (filename defaults
        to the model name if not specified) and all associated assets (e.g. meshes)."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        mjcf.export_with_assets(self.mjcf_root, str(output_dir), xml_filename)
