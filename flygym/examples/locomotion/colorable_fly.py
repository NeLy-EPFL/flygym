import numpy as np
from dm_control import mjcf
from flygym import Fly


class ColorableFly(Fly):
    """
    A wrapper around the Fly class that facilitates the recoloring of
    specific segments. This is useful for, as an example, recoloring parts
    of the leg depending on the activation of specific correction rules.

    This class is necessary because the leg segments would otherwise not be
    colored as intended: "textures are applied in GL_MODULATE mode, meaning
    that the texture color and the color specified here are multiplied
    component-wise" as mentioned in the MuJoCo documentation. This class
    overrides the impact of the default texture on the resulting final
    color. See
    https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-material-rgba
    """

    def __init__(self, recolor_types=("femur", "tibia"), **kwargs):
        self.default_segment_rgba = {}
        self.recolor_types = recolor_types
        super().__init__(**kwargs)

    def _set_geom_colors(self):
        for type_, specs in self.config["appearance"].items():
            # Define texture and material
            recolor = type_ in self.recolor_types
            rgba = specs["material"]["rgba"]

            if specs["texture"] is not None:
                rgb1 = specs["texture"]["rgb1"]
                rgb2 = specs["texture"]["rgb2"]
                self.model.asset.add(
                    "texture",
                    name=f"{type_}_texture",
                    builtin=specs["texture"]["builtin"],
                    mark="random",
                    width=specs["texture"]["size"],
                    height=specs["texture"]["size"],
                    random=specs["texture"]["random"],
                    rgb1=(1, 1, 1) if recolor else rgb1,
                    rgb2=(1, 1, 1) if recolor else rgb2,
                    markrgb=specs["texture"]["markrgb"],
                )

                if recolor:
                    rgba = (*np.mean([rgb1, rgb2], axis=0), rgba[3])

            self.model.asset.add(
                "material",
                name=f"{type_}_material",
                texture=f"{type_}_texture" if specs["texture"] is not None else None,
                rgba=rgba,
                specular=0.0,
                shininess=0.0,
                reflectance=0.0,
                texuniform=True,
            )
            # Apply to geoms
            for segment in specs["apply_to"]:
                self.default_segment_rgba[segment] = rgba
                geom = self.model.find("geom", segment)
                if geom is None:
                    geom = self.model.find("geom", f"{segment}")
                geom.material = f"{type_}_material"

    def change_segment_color(self, physics: mjcf.Physics, segment: str, color=None):
        """Change the color of a segment of the fly.

        Parameters
        ----------
        physics : mjcf.Physics
            The physics object of the simulation.
        segment : str
            The name of the segment to change the color of.
        color : tuple[float, float, float, float]
            Target color as RGBA values normalized to [0, 1].
        """
        if not color:
            color = self.default_segment_rgba[segment]

        physics.named.model.geom_rgba[f"{self.name}/{segment}"][: len(color)] = color
