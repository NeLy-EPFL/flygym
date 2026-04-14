import xml.etree.ElementTree as ET
from pathlib import Path
import yaml

# density/=1000, viscosity/=10, forcerange*=10, pos*=10, gravity*=10, gainprm*=10, biasprm*=10, etc.
units_mapping = {
    "pos": 10,
    "size": 10,
    "fromto": 10,
    "density": 1e-3,
    "viscosity": 1e-1,
    "forcerange": 10,
    "gainprm": 10,
    "biasprm": 10,
}

side_mapping = {"left":"l", "right":"r"}
leg_mapping = {"T1":"f", "T2":"m", "T3":"h"}

def map_flybody_bname_to_flygym_bname(bname):
    s_bname = bname.split("_")
    if "femur" in s_bname:
        s_bname[s_bname.index("femur")] = "trochanterfemur"
    match len(s_bname):
        case 1:
            if bname == "abdomen":
                return "c_abdomen1"
            else:
                return f"c_{bname}"
        case 2:
            if s_bname[1] in side_mapping:
                flygym_side = side_mapping[s_bname[1]]
                return f"{flygym_side}_{s_bname[0]}"
            elif s_bname[1].isdigit():
                # TO DO DEAL WITH DIFFERENT AMOUNT OF ABD SEGMENTS
                return f"c_{''.join(s_bname)}"
            else:
                raise ValueError(f"Unexpected segmented name: {s_bname}")
        case 3:
            if s_bname[0] == "tarsus":
                s_bname[0] = "tarsus1"
            if s_bname[1] in leg_mapping and s_bname[2] in side_mapping:
                flygym_side = side_mapping[s_bname[2]]
                flygym_leg = leg_mapping[s_bname[1]]
                return f"{flygym_side}{flygym_leg}_{s_bname[0]}"
            else:
                raise ValueError(f"Unexpected segmented name: {s_bname}")
        case 4:
            if s_bname[2] in leg_mapping and s_bname[3] in side_mapping and s_bname[1] == "claw":
                flygym_side = side_mapping[s_bname[3]]
                flygym_leg = leg_mapping[s_bname[2]]
                return f"{flygym_side}{flygym_leg}_{s_bname[1]}"
            elif s_bname[1] in leg_mapping and s_bname[2].isdigit() and s_bname[3] in side_mapping:
                flygym_side = side_mapping[s_bname[3]]
                flygym_leg = leg_mapping[s_bname[1]]
                return f"{flygym_side}{flygym_leg}_{s_bname[0]}{s_bname[2]}"
            else:
                raise ValueError(f"Unexpected segmented name: {s_bname}")


class NiceDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def _represent_list(dumper, data):
    # Force lists onto a single line: [1, 2, 3]
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


NiceDumper.add_representer(list, _represent_list)


def _split_whitespace_to_list(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return []
        if any(ch.isspace() for ch in stripped):
            return stripped.split()
        return stripped
    return value


def _first_or_value(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _scale_numeric_value(value, scale):
    if isinstance(value, (int, float)):
        scaled_value = value * scale
    else:
        try:
            scaled_value = float(value) * scale
        except (TypeError, ValueError):
            return value

    if float(scaled_value).is_integer():
        return str(int(scaled_value))
    return str(scaled_value)


def _parse_and_scale_attr(attr_name, value):
    parsed_value = _split_whitespace_to_list(value)
    scale = units_mapping.get(attr_name)
    if scale is None:
        return parsed_value

    if isinstance(parsed_value, list):
        return [_scale_numeric_value(item, scale) for item in parsed_value]

    return _scale_numeric_value(parsed_value, scale)


def _write_yaml_file(path, data):
    with open(path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            data,
            yaml_file,
            Dumper=NiceDumper,
            sort_keys=False,
            default_flow_style=False,
            indent=2,
            width=10_000,
            allow_unicode=True,
        )


def _is_excluded_geom(geom_name):
    return (
        "collision" in geom_name
        or "fluid" in geom_name
        or "inertial" in geom_name
    )


def parse_xml_to_rig(xml_path, yaml_path):
    all_geom_suffixes = {}
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    default_lookup = build_effective_default_lookup(root)
    wing_inertial_mass = (
        default_lookup.get("wing-inertial", {}).get("geom", {}).get("mass")
    )

    def _get_effective_geom_defaults(body_childclass, geom_class):
        resolved_geom_defaults = {}

        if "__root__" in default_lookup:
            resolved_geom_defaults.update(default_lookup["__root__"].get("geom", {}))

        if body_childclass in default_lookup:
            resolved_geom_defaults.update(default_lookup[body_childclass].get("geom", {}))

        if geom_class in default_lookup:
            resolved_geom_defaults.update(default_lookup[geom_class].get("geom", {}))

        return resolved_geom_defaults

    rigging_parsed = {}

    def parse_body_recursive(body, inherited_childclass):
        body_childclass = body.get("childclass", inherited_childclass)
        flybody_name = body.get("name")
        flygym_name = map_flybody_bname_to_flygym_bname(flybody_name)
        selected_data = {
            k: _parse_and_scale_attr(k, v)
            for k, v in body.attrib.items()
            if k in ["pos", "quat"]
        }
        if not "pos" in selected_data:
            selected_data["pos"] = _parse_and_scale_attr("pos", "0 0 0")
        if not "quat" in selected_data:
            selected_data["quat"] = _parse_and_scale_attr("quat", "1 0 0 0")

        selected_data["geoms"] = {}

        for child_geom in body.findall("geom"):

            if _is_excluded_geom(child_geom.get("name")):
                # We want collision to happen with real geom not capsules
                # We do not care about flight
                continue

            geom_selected_data = {}
            geom_defaults = _get_effective_geom_defaults(
                body_childclass, child_geom.get("class")
            )
            for default_param in ["density", "mass"]:
                if default_param in geom_defaults:
                    geom_selected_data[default_param] = _parse_and_scale_attr(
                        default_param, geom_defaults[default_param]
                    )

            for k, v in child_geom.attrib.items():
                # if k in ["pos", "quat", "type", "size", "fromto"]:
                #     geom_selected_data[k] = v
                if k == "mesh":
                    geom_selected_data[k] = translate_mesh_name(v, all_geom_suffixes)
                elif k == "name" or k == "material" or "class" in k:
                    # We want to keep the original name of the geom for the visuals, but we will use the flygym name for the rigging
                    continue
                else:
                    geom_selected_data[k] = _parse_and_scale_attr(k, v)

            geom_name = child_geom.get("name") or ""
            is_wing_brown = body_childclass == "wing" and geom_name.endswith("_brown")
            is_wing_membrane = body_childclass == "wing" and geom_name.endswith("_membrane")

            if is_wing_membrane and wing_inertial_mass is not None:
                # Keep inertial geoms out of rigging while transferring their mass.
                geom_selected_data["mass"] = _parse_and_scale_attr(
                    "mass", wing_inertial_mass
                )

            if is_wing_brown:
                geom_selected_data["mass"] = _parse_and_scale_attr("mass", "0")
                geom_selected_data["density"] = _parse_and_scale_attr("density", "0")

            # If mass is explicitly/effectively set, keep mass and skip density.
            if "mass" in geom_selected_data and not is_wing_brown:
                geom_selected_data.pop("density", None)

            selected_data["geoms"][child_geom.get("name")] = geom_selected_data

        rigging_parsed[flygym_name] = selected_data
        rigging_parsed[flygym_name]["flybody_name"] = flybody_name

        for child_body in body.findall("body"):
            parse_body_recursive(child_body, body_childclass)

    worldbody = root.find("worldbody")
    for top_level_body in worldbody.findall("body"):
        parse_body_recursive(top_level_body, None)
    
    # print(rigging_parsed)

    _write_yaml_file(yaml_path, rigging_parsed)

    _write_yaml_file(yaml_path.with_name("flybody_all_geom_suffixes.yaml"), all_geom_suffixes)

def translate_mesh_name(mesh_name, all_segment_suffixes):
    meshes_suffixes = [ 
        "collision", "collision2", "black", "red", "ocelli",
        "bristle-brown", "lower", "membrane", "brown", "body"
    ]
    bname = mesh_name
    for suffix in meshes_suffixes:
        if suffix in bname:
            bname = bname.replace(f"_{suffix}", "")
    if bname == mesh_name:
        full_suffix = "body"
    else:
        full_suffix = mesh_name.replace(f"{bname}_", "")
    flygym_name = map_flybody_bname_to_flygym_bname(bname)
    if not flygym_name in all_segment_suffixes:
        all_segment_suffixes[flygym_name] = [full_suffix]
    else:
        all_segment_suffixes[flygym_name].append(full_suffix)
    new_mesh_name = f"{flygym_name}_{full_suffix}" if full_suffix else flygym_name
    return new_mesh_name

def parse_meshes(flybody_mesh_dir, mesh_dir):
    flybody_mesh_dir = Path(flybody_mesh_dir)
    mesh_dir = Path(mesh_dir)
    # remove all existing meshes in the target directory
    for existing_mesh in mesh_dir.glob("*.obj"):
        existing_mesh.unlink()
    for mesh_path in flybody_mesh_dir.glob("*.obj"):
        mesh_name = mesh_path.stem
        new_mesh_name = translate_mesh_name(mesh_name, {})
        if "None" in new_mesh_name:
            print(f"Warning: mesh {mesh_name} has no suffix, skipping")
            continue
        target_path = mesh_dir / f"{new_mesh_name}.obj"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(mesh_path.read_bytes())

def parse_visuals(xml_path, yaml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # retrieve all materials with parameters
    parsed_visuals = {}
    for material in root.findall("asset/material"):
        material_name = material.get("name")
        parsed_visuals[material_name] = {}
        material_params = {
            k: _split_whitespace_to_list(v)
            for k, v in material.attrib.items()
            if k != "name"
        }
        parsed_visuals[material_name]["apply_to"] = f"*_{material_name}"
        parsed_visuals[material_name]["material"] = material_params
    
    _write_yaml_file(yaml_path, parsed_visuals)


dof_mapping = {
    "twist": "roll",
    "extend": "pitch",
    "abduct": "yaw",
    "roll": "roll",
    "pitch": "pitch",
    "yaw": "yaw",
}


def build_effective_default_lookup(root):
    default_root = root.find("default")
    if default_root is None:
        raise ValueError("No <default> section found in the XML model.")

    effective_defaults = {}

    # MuJoCo allows unnamed defaults directly under <default>. These are the true
    # base defaults inherited by all named classes.
    root_inherited_params = {}
    for child in default_root:
        if child.tag == "default":
            continue
        root_inherited_params.setdefault(child.tag, {})
        root_inherited_params[child.tag].update(
            {k: _split_whitespace_to_list(v) for k, v in child.attrib.items()}
        )

    def recurse(default_element, inherited_params):
        current_params = {
            tag: params.copy() for tag, params in inherited_params.items()
        }

        for child in default_element:
            if child.tag == "default":
                recurse(child, current_params)
                continue

            current_params.setdefault(child.tag, {})
            current_params[child.tag].update(
                {k: _split_whitespace_to_list(v) for k, v in child.attrib.items()}
            )

        class_name = default_element.get("class")
        if class_name is not None:
            effective_defaults[class_name] = current_params

    for top_level_default in default_root.findall("default"):
        recurse(top_level_default, root_inherited_params)

    if root_inherited_params:
        effective_defaults["__root__"] = root_inherited_params

    return effective_defaults


def _collect_class_apply_targets(worldbody):
    """Collect where each class is used in the kinematic tree.

    Returns:
        dict[class_name, set[str]] mapping to flygym joint names only.
    """
    class_to_targets = {}

    def add_target(class_name, target):
        if class_name is None:
            return
        class_to_targets.setdefault(class_name, set()).add(target)

    def recurse(parent_body, inherited_childclass=None):
        for child_body in parent_body.findall("body"):
            body_childclass = child_body.get("childclass", inherited_childclass)

            for joint in child_body.findall("joint"):
                joint_class = joint.get("class", body_childclass)
                if joint_class is not None:
                    joint_name = get_flygym_jointname(parent_body, child_body, joint)
                    add_target(joint_class, joint_name)

            recurse(child_body, body_childclass)

    recurse(worldbody)
    return class_to_targets


def _collect_default_class_hierarchy(root):
    """Collect parent and local (non-inherited) params for each default class."""
    default_root = root.find("default")
    if default_root is None:
        raise ValueError("No <default> section found in the XML model.")

    hierarchy = {}

    def recurse(default_element, parent_class):
        class_name = default_element.get("class")
        local_params = {}
        for child in default_element:
            if child.tag == "default":
                continue
            local_params.setdefault(child.tag, {})
            local_params[child.tag].update(
                {k: _split_whitespace_to_list(v) for k, v in child.attrib.items()}
            )

        if class_name is not None:
            hierarchy[class_name] = {
                "parent": parent_class,
                "local": local_params,
            }
            next_parent = class_name
        else:
            next_parent = parent_class

        for child in default_element.findall("default"):
            recurse(child, next_parent)

    for top_level_default in default_root.findall("default"):
        recurse(top_level_default, None)

    return hierarchy


def _is_ignored_actuator_class(class_name):
    lowered = class_name.lower()
    return "collision" in lowered or "adhesion" in lowered


def _clean_actuator_tag_config(tag, tag_cfg, ignore_ctrlrange):
    cleaned = tag_cfg.copy()
    if ignore_ctrlrange and tag == "general":
        cleaned.pop("ctrlrange", None)
    return cleaned


def _scale_actuator_tag_config(tag_cfg):
    return {
        key: _parse_and_scale_attr(key, value)
        for key, value in tag_cfg.items()
    }


def _has_meaningful_local_actuation(class_name, hierarchy, actuator_tags, ignore_ctrlrange):
    class_info = hierarchy.get(class_name, {})
    local_params = class_info.get("local", {})
    for tag in actuator_tags:
        if tag not in local_params:
            continue
        cleaned = _clean_actuator_tag_config(tag, local_params[tag], ignore_ctrlrange)
        if cleaned:
            return True
    return False


def _resolve_representative_class(class_name, hierarchy, actuator_tags, ignore_ctrlrange):
    """Map a class to the nearest ancestor with meaningful actuator defaults."""
    current = class_name
    while current is not None:
        if _is_ignored_actuator_class(current):
            return None
        if _has_meaningful_local_actuation(
            current, hierarchy, actuator_tags, ignore_ctrlrange
        ):
            return current
        current = hierarchy.get(current, {}).get("parent")
    return None


def _merge_equivalent_actuator_groups(groups):
    """Merge classes that resolve to exactly the same actuator/default params."""
    signature_to_group = {}
    for class_name, cfg in groups.items():
        payload = {k: v for k, v in cfg.items() if k != "apply_to"}
        signature = yaml.safe_dump(payload, sort_keys=True)
        if signature not in signature_to_group:
            signature_to_group[signature] = {
                "classes": [class_name],
                "apply_to": cfg.get("apply_to", []),
                **payload,
            }
            continue

        signature_to_group[signature]["classes"].append(class_name)
        existing_apply = signature_to_group[signature].get("apply_to", [])
        new_apply = cfg.get("apply_to", [])
        if isinstance(existing_apply, str):
            existing_apply = [existing_apply]
        if isinstance(new_apply, str):
            new_apply = [new_apply]
        merged_apply = sorted(set(existing_apply + new_apply))
        signature_to_group[signature]["apply_to"] = merged_apply

    merged = {}
    for entry in signature_to_group.values():
        classes = sorted(entry.pop("classes"))
        out_name = classes[0] if len(classes) == 1 else "-".join(classes)
        apply_to = entry.get("apply_to", [])
        if isinstance(apply_to, list):
            apply_to = sorted(set(apply_to))
            if len(apply_to) == 1:
                apply_to = apply_to[0]
        entry["apply_to"] = apply_to
        merged[out_name] = entry

    return merged


def parse_actuators(xml_path, yaml_path, ignore_ctrlrange=True, merge_equivalent=True):
    """Parse inherited actuator defaults from MuJoCo defaults tree.

    The output groups are class-based defaults containing actuator-related tags
    (e.g. general, adhesion, motor, position, ...), plus an apply_to section
    inferred from body/joint class usage.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    default_lookup = build_effective_default_lookup(root)
    class_to_targets = _collect_class_apply_targets(root.find("worldbody"))
    class_hierarchy = _collect_default_class_hierarchy(root)

    actuator_tags = {
        "general",
        "motor",
        "position",
        "velocity",
        "intvelocity",
        "damper",
        "cylinder",
        "muscle",
    }

    representative_to_targets = {}
    for class_name, targets in class_to_targets.items():
        representative = _resolve_representative_class(
            class_name, class_hierarchy, actuator_tags, ignore_ctrlrange
        )
        if representative is None:
            continue
        representative_to_targets.setdefault(representative, set()).update(targets)

    parsed = {}
    for class_name, targets in representative_to_targets.items():
        if class_name == "__root__" or class_name not in default_lookup:
            continue
        class_params = default_lookup[class_name]

        group_cfg = {}
        for tag in actuator_tags:
            if tag not in class_params:
                continue

            tag_cfg = _clean_actuator_tag_config(
                tag, class_params[tag], ignore_ctrlrange
            )

            if tag_cfg:
                group_cfg[tag] = _scale_actuator_tag_config(tag_cfg)

        if not group_cfg:
            continue

        apply_to = sorted(targets)
        if len(apply_to) == 1:
            group_cfg["apply_to"] = apply_to[0]
        else:
            group_cfg["apply_to"] = apply_to

        parsed[class_name] = group_cfg

    if merge_equivalent:
        parsed = _merge_equivalent_actuator_groups(parsed)

    _write_yaml_file(yaml_path, parsed)

def get_flygym_jointname(parent_body, child_body, joint):
    parent_flygym_bn = map_flybody_bname_to_flygym_bname(parent_body.get("name"))
    child_flygym_bn = map_flybody_bname_to_flygym_bname(child_body.get("name"))
    tokens = []
    joint_name = joint.get("name")
    if joint_name:
        tokens.extend(joint_name.split("_"))

    joint_class = joint.get("class")
    if joint_class:
        tokens.extend(joint_class.split("_"))

    got_dof = [token in dof_mapping for token in tokens]
    if any(got_dof):
        flybody_dof = tokens[got_dof.index(True)]
        dof = dof_mapping[flybody_dof]
    else:
        dof = "pitch"
        # a^get axis and assert 1 0 0
        axis = joint.get("axis")
        if axis is not None:
            assert axis == "1 0 0", f"Expected joint axis '1 0 0' for default dof inference, got '{axis}' for joint '{joint_name}' between '{parent_body.get('name')}' and '{child_body.get('name')}'. Please specify a dof explicitly in the XML or ensure the axis is correct for inference."
        else:
            print(f"Warning: No axis specified for joint '{joint_name}' between '{parent_body.get('name')}' and '{child_body.get('name')}'. Defaulting to '1 0 0' for pitch dof inference. Check manually in defaults that axis is 1 0 0.")
    flygym_jointname = f"{parent_flygym_bn}-{child_flygym_bn}-{dof}"
    return flygym_jointname

def add_class_params(default_lookup, tag, class_name, accumulated_params):
    class_params = default_lookup.get(class_name)
    if class_params is None:
        raise ValueError(f"Class {class_name} not found in defaults")
    accumulated_params.update(class_params.get(tag, {}))


def recursive_accumulation_joint_params(
    parent_body, accumulated_params, default_lookup
):
    all_joints = {}
    for child_body in parent_body.findall("body"):
        # 1. Create a copy so sibling branches don't pollute each other's parameters
        current_accumulated = accumulated_params.copy()
        
        child_class = child_body.get("childclass")
        if child_class is not None:
            # Overwrite with childclass defaults
            add_class_params(default_lookup, "joint", child_class, current_accumulated)
            
        for joint in child_body.findall("joint"):
            # 2. Start from the accumulated parameters (inherited from childclasses)
            joint_params = current_accumulated.copy()
            
            # Add defaults from the explicit joint class, if present
            if joint.get("class") is not None:
                add_class_params(
                    default_lookup, "joint", joint.get("class"), joint_params
                )
            # 3. Finally, apply explicit joint attributes. This ensures they have 
            # highest priority and overwrite class defaults.
            selected_joint_attribs = {k: v for k, v in joint.attrib.items() if k not in ["name", "class"]}
            selected_joint_attribs = {
                k: _split_whitespace_to_list(v)
                for k, v in selected_joint_attribs.items()
            }
            joint_params.update(selected_joint_attribs)
            
            flygym_jointname = get_flygym_jointname(parent_body, child_body, joint)
            all_joints[flygym_jointname] = joint_params
            
        # Traverse deeper using the correctly scoped parameters
        child_joints = recursive_accumulation_joint_params(
            child_body, current_accumulated, default_lookup
        )
        all_joints.update(child_joints)
        
    return all_joints


def _group_joint_params_for_yaml(all_joints_parsed):
    """Group shared joint parameters and split per-joint ranges/springrefs.

    Returns a dict with two top-level sections:
      - params: grouped shared joint params with apply_to targets
      - ranges: per-joint range/springref overrides
    """
    signature_to_group = {}

    for joint_name, joint_params in all_joints_parsed.items():
        shared_params = {
            k: v
            for k, v in joint_params.items()
            if k not in {"range", "springref", "axis", "group"}
        }
        signature = yaml.safe_dump(shared_params, sort_keys=True)

        if signature not in signature_to_group:
            signature_to_group[signature] = {
                "apply_to": [],
                **shared_params,
            }
        signature_to_group[signature]["apply_to"].append(joint_name)

    grouped_params = {}
    sorted_groups = sorted(signature_to_group.items(), key=lambda item: item[0])
    for i, (_, group_cfg) in enumerate(sorted_groups, start=1):
        apply_to = sorted(group_cfg["apply_to"])
        group_cfg["apply_to"] = apply_to[0] if len(apply_to) == 1 else apply_to
        grouped_params[f"group_{i:03d}"] = group_cfg

    ranges = {}
    for joint_name, joint_params in sorted(all_joints_parsed.items()):
        range_cfg = {}
        if "range" in joint_params:
            range_cfg["range"] = joint_params["range"]
        if "springref" in joint_params:
            range_cfg["springref"] = joint_params["springref"]
        if range_cfg:
            ranges[joint_name] = range_cfg

    return {"params": grouped_params, "ranges": ranges}


def parse_joints(xml_path, joint_yaml_path, pose_yaml_path, kin_order):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    default_lookup = build_effective_default_lookup(root)
    all_joints_parsed = recursive_accumulation_joint_params(
        root.find("worldbody"), {}, default_lookup
    )
    grouped_joints_parsed = _group_joint_params_for_yaml(all_joints_parsed)
    neutral_pose_parsed = {
        "angle_unit": "radian",
        "axis_order": kin_order.split("_"),
        "joint_angles": {}
    }
    for joint_name, joint_params in all_joints_parsed.items():
        if "springref" in joint_params:
            springref = _first_or_value(joint_params["springref"])
            if springref is not None:
                neutral_pose_parsed["joint_angles"][joint_name] = float(springref)
    
    _write_yaml_file(joint_yaml_path, grouped_joints_parsed)

    _write_yaml_file(pose_yaml_path, neutral_pose_parsed)
    
    return

def parse_globals(xml_path, yaml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    parsed_globals = {}
    for tag in ["compiler", "option", "size"]:
        element = root.find(tag)
        if element is not None:
            parsed_globals[tag] = {
                k: _split_whitespace_to_list(v)
                for k, v in element.attrib.items()
            }

    parsed_globals["compiler"]["fusestatic"] = "true"
    parsed_globals["statistic"] = {"extent": "5"}
    parsed_globals["visual"] = {"headlight":{
        "ambient": "0.5 0.5 0.5",
        "diffuse": "0.6 0.6 0.6",
        "specular": "0. 0. 0.",
    }}

    _write_yaml_file(yaml_path, parsed_globals)

if __name__ == "__main__":
    out_dir = Path("src/flygym/assets/model/flybody")
    flybody_xml = Path("/Users/stimpfli/Desktop/mujoco_menagerie/flybody/fruitfly.xml")

    rigging_flybody_yaml = out_dir / "flybody_rigging.yaml"
    parse_xml_to_rig(flybody_xml, rigging_flybody_yaml)
    
    visuals_flybody_yaml = out_dir / "flybody_visuals.yaml"
    parse_visuals(flybody_xml, visuals_flybody_yaml)

    actuators_flybody_yaml = out_dir / "flybody_actuators.yaml"
    parse_actuators(flybody_xml, actuators_flybody_yaml)
    
    flybody_meshes_dir = flybody_xml.parent / "assets"
    meshes_out_dir = out_dir / "meshes"
    parse_meshes(flybody_meshes_dir, meshes_out_dir)
    
    joint_yaml_path = out_dir / "flybody_joints.yaml"
    kin_order = "yaw_roll_pitch"
    flight_pose_path = out_dir / f"pose/flight/{kin_order}.yaml"
    # the neutral pose is actually set for flight (real neutral pose is all 0)
    parse_joints(flybody_xml, joint_yaml_path, flight_pose_path, kin_order)
    neutral_pose_path = out_dir / f"pose/neutral/{kin_order}.yaml"
    neutral_pose_parsed = {
        "angle_unit": "radian",
        "axis_order": kin_order.split("_"),
        "joint_angles": {
            "c_thorax-l_wing-yaw": 1.5,
            "c_thorax-l_wing-roll": 0.7,
            "c_thorax-l_wing-pitch": -1.0,}
    }
    _write_yaml_file(neutral_pose_path, neutral_pose_parsed)

    globals_flybody_yaml = out_dir / "flybody_mujoco_globals.yaml"
    parse_globals(flybody_xml, globals_flybody_yaml)
