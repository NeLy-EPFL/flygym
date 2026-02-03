from abc import ABC, abstractmethod
from pathlib import Path
from os import PathLike
from collections.abc import Sequence
from numbers import Number
from tempfile import TemporaryDirectory

import mujoco
import dm_control.mjcf as mjcf

from flygym.utils.math import Rotation3D


class BaseCompositionElement(ABC):
    @abstractmethod
    def get_mjcf_root(self) -> mjcf.RootElement:
        """Returns the MJCF root element representing the fly."""
        pass

    def compile(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        with TemporaryDirectory(delete=True) as tempdir:
            tempdir = Path(tempdir)
            self.save_xml_with_assets(tempdir, out_file_name="model.xml")
            mj_model = mujoco.MjModel.from_xml_path(str(tempdir / "model.xml"))
            mj_data = mujoco.MjData(mj_model)
            return mj_model, mj_data

    def save_xml_with_assets(
        self, output_dir: PathLike, out_file_name: str = None
    ) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        mjcf.export_with_assets(
            self.get_mjcf_root(), str(output_dir), out_file_name=out_file_name
        )


class BaseFly(BaseCompositionElement, ABC):
    pass


class BaseWorld(BaseCompositionElement, ABC):
    @abstractmethod
    def spawn_fly(
        self,
        fly: BaseFly,
        spawn_position: Sequence[Number],
        spawn_rotation: Rotation3D,
        **kwargs,
    ) -> None:
        """Adds a fly to the world at the specified position and orientation."""
        pass
