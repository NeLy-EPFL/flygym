from abc import ABC, abstractmethod
from pathlib import Path
from os import PathLike
from tempfile import TemporaryDirectory

import mujoco
import dm_control.mjcf as mjcf


class BaseCompositionElement(ABC):
    @property
    @abstractmethod
    def mjcf_root(self) -> mjcf.RootElement:
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
            self.mjcf_root, str(output_dir), out_file_name=out_file_name
        )
