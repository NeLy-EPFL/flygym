import os
import shutil
import contextlib
from abc import ABC, abstractmethod
from pathlib import Path
from os import PathLike

import mujoco as mj

__all__ = ["BaseCompositionElement"]


class BaseCompositionElement(ABC):
    """Base class for composable elements in the MuJoCo model, providing common
    functionality such as compiling to MuJoCo model/data and exporting."""

    @property
    @abstractmethod
    def mjcf_root(self) -> mj.MjSpec:
        """The root MjSpec of this composition element."""
        pass

    def compile(self) -> tuple[mj.MjModel, mj.MjData]:
        """Compile the MjSpec into MuJoCo MjModel and MjData objects. This is where
        the model edited via `mujoco.MjSpec` is "handed off" to `mujoco` for
        simulation. Things like the ordering of generalized coordinates (qpos) are
        determined here.

        We compile a copy of the spec rather than the spec itself: with
        `compiler/fusestatic` enabled, compiling mutates the spec in place by fusing
        static bodies into their parent. During composition (e.g. before a fly is
        given a free joint by the world) the root body is still static, so compiling
        the live spec would destroy it and invalidate the element references FlyGym
        holds. Compiling a copy keeps the editable spec intact while producing an
        equivalent compiled model."""
        model = self.mjcf_root.copy().compile()
        data = mj.MjData(model)
        return model, data

    def save_xml_with_assets(
        self, output_dir: PathLike, xml_filename: str = None
    ) -> None:
        """Export the model to a directory, including an XML file (filename defaults
        to the model name if not specified) and all associated assets (e.g. meshes).

        File-based assets are copied next to the XML and their references are
        relativized so the exported model is self-contained (e.g. loadable via
        `mj_loadXML` in the browser).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Operate on a copy: relativizing asset paths and clearing the asset search
        # dirs mutates the spec, which would break later compile() calls on the live
        # model (it would no longer find its meshes).
        spec = self.mjcf_root.copy()

        # Copy file-based assets (meshes, file textures) next to the XML and rewrite
        # their `file` attributes to bare filenames so the export is self-contained.
        asset_collections = [spec.meshes, spec.textures]
        for collection in asset_collections:
            for asset in collection:
                src = getattr(asset, "file", None)
                if not src:
                    continue
                src_path = Path(src)
                dst_path = output_dir / src_path.name
                if src_path.resolve() != dst_path.resolve():
                    shutil.copy(src_path, dst_path)
                asset.file = src_path.name
        # Assets now sit alongside the XML, so clear the asset search dirs. The
        # emitted XML then references assets by bare filename, resolved relative to
        # the XML's own directory when later loaded with `mj_loadXML`.
        spec.meshdir = ""
        spec.texturedir = ""

        # `to_xml()` validates assets by loading them, resolving bare filenames
        # against the working directory, so serialize from within `output_dir`
        # (where the assets were just copied).
        xml_filename = xml_filename or f"{spec.modelname}.xml"
        with _working_directory(output_dir):
            xml_string = spec.to_xml()
        (output_dir / xml_filename).write_text(xml_string)


@contextlib.contextmanager
def _working_directory(path: PathLike):
    """Temporarily change the process working directory."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
