from abc import ABC, abstractmethod
from pathlib import Path
from os import PathLike

import mujoco
import dm_control.mjcf as mjcf


class BaseCompositionElement(ABC):
    @property
    @abstractmethod
    def mjcf_root(self) -> mjcf.RootElement:
        """Returns the MJCF root element representing the fly."""
        pass

    def compile(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        physics = mjcf.Physics.from_mjcf_model(self.mjcf_root)
        return physics.model._model, physics.data._data

    def save_xml_with_assets(
        self, output_dir: PathLike, xml_filename: str = None
    ) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        mjcf.export_with_assets(self.mjcf_root, str(output_dir), xml_filename)
