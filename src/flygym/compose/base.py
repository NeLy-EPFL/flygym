from abc import ABC, abstractmethod
from pathlib import Path
from os import PathLike

import mujoco as mj
import dm_control.mjcf as mjcf

__all__ = ["BaseCompositionElement"]


class BaseCompositionElement(ABC):
    """Base class for composable elements in the MuJoCo model, providing common
    functionality such as compiling to MuJoCo model/data and exporting."""

    @property
    @abstractmethod
    def mjcf_root(self) -> mjcf.RootElement:
        pass

    def compile(self) -> tuple[mj.MjModel, mj.MjData]:
        """Compile the MJCF model into MuJoCo MjModel and MjData objects. This is where
        `dm_control.mjcf` "hands off" the model that it composes to `mujoco` for
        simulation. Things like the ordering of generalized coordinates (qpos) are
        determined here."""
        # physics = mjcf.Physics.from_mjcf_model(self.mjcf_root)
        # return physics.model._model, physics.data._data

        # Quick hack for relaxing MuJoCo/dm_control version dependency:
        # We're using pre-release versions of MJWarp for the lateste features. These are
        # not compatible with the latest stable release of dm_control, which is what we
        # use for MJCF composition. The workaround is to not use dm_control's Physics
        # wrapper at all, and instead export the MJCF model to XML and then load it
        # directly with mujoco.

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self.save_xml_with_assets(tmpdir, "model.xml")
            mj_model = mj.MjModel.from_xml_path((tmpdir / "model.xml").as_posix())
            mj_data = mj.MjData(mj_model)

            return mj_model, mj_data

    def save_xml_with_assets(
        self, output_dir: PathLike, xml_filename: str = None
    ) -> None:
        """Export the MJCF model to a directory, including a XML file (filename defaults
        to the model name if not specified) and all associated assets (e.g. meshes)."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        mjcf.export_with_assets(self.mjcf_root, str(output_dir), xml_filename)
