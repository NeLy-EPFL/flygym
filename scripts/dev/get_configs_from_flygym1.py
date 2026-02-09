import xml.etree.ElementTree as ET
import math
from pathlib import Path

import yaml

import flygym
from flygym.utils.api1to2 import BODY_NAMES_OLD2NEW


def _round_to_sigfigs(x: float, sigfigs: int = 3) -> float:
    if x == 0:
        return 0.0
    return round(x, sigfigs - int(math.floor(math.log10(abs(x)))) - 1)


def get_physical_params_from_legacy_mjcf(mjcf_path: Path):
    mjcf_tree = ET.parse(mjcf_path)
    mjcf_root = mjcf_tree.getroot()

    pose_by_body = {}
    for body in mjcf_root.findall(".//body"):
        name = body.attrib["name"]
        if name == "FlyBody":
            continue  # skip the "virtual" root body - it doesn't anatomically exist
        pos = [_round_to_sigfigs(float(x)) for x in body.attrib["pos"].split()]
        quat = [_round_to_sigfigs(float(x)) for x in body.attrib["quat"].split()]
        pose_by_body[name] = {"pos": pos, "quat": quat}

    mass_by_body = {}
    for geom in mjcf_root.findall(".//geom"):
        name = geom.attrib["name"]
        mass = _round_to_sigfigs(float(geom.attrib["mass"]))
        mass_by_body[name] = {"mass": mass}

    assert set(pose_by_body.keys()) == set(mass_by_body.keys())
    return {k: {**pose_by_body[k], **mass_by_body[k]} for k in pose_by_body.keys()}


def save_pretty_yaml(data_to_save: dict, output_path: Path):
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    def represent_list(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    NoAliasDumper.add_representer(list, represent_list)
    yaml_str = yaml.dump(data_to_save, Dumper=NoAliasDumper, sort_keys=False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(yaml_str)


if __name__ == "__main__":
    model_dir = flygym.assets_dir / "model"
    ref_mjcf_path = model_dir / "legacy/flygym1_deepfly3d_rollyawpitch.xml"
    output_path = model_dir / "rigging.yaml"

    physical_params_by_body = get_physical_params_from_legacy_mjcf(ref_mjcf_path)
    assert set(physical_params_by_body.keys()) == BODY_NAMES_OLD2NEW.keys()

    params_by_body = {
        new_name: physical_params_by_body[old_name]
        for old_name, new_name in BODY_NAMES_OLD2NEW.items()
    }
    save_pretty_yaml(params_by_body, output_path)
