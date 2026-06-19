"""Unit tests for ``flygym.flybody.parse_flybody``.

``parse_flybody`` is the offline tool that converts the flybody MuJoCo XML into
the flygym YAML rig/joint/actuator/visual files (run via its ``__main__``). It
is not imported at runtime, but it is large and logic-heavy, so these tests pin
down the pure helpers (name mapping, unit scaling, mesh-suffix extraction,
actuator-default resolution) plus the XML-driven parsers exercised against a
small synthetic MuJoCo model.
"""

import xml.etree.ElementTree as ET

import pytest
import yaml

from flygym.flybody import parse_flybody as pf


# ---------------------------------------------------------------------------
# map_flybody_bname_to_flygym_bname
# ---------------------------------------------------------------------------


class TestMapBodyName:
    def test_single_token_abdomen_special_case(self):
        assert pf.map_flybody_bname_to_flygym_bname("abdomen") == "c_abdomen1"

    def test_single_token_central(self):
        assert pf.map_flybody_bname_to_flygym_bname("thorax") == "c_thorax"
        assert pf.map_flybody_bname_to_flygym_bname("head") == "c_head"

    def test_two_token_with_side(self):
        assert pf.map_flybody_bname_to_flygym_bname("wing_left") == "l_wing"
        assert pf.map_flybody_bname_to_flygym_bname("wing_right") == "r_wing"

    def test_two_token_with_digit(self):
        assert pf.map_flybody_bname_to_flygym_bname("tergite_3") == "c_tergite3"

    def test_two_token_invalid_raises(self):
        with pytest.raises(ValueError):
            pf.map_flybody_bname_to_flygym_bname("foo_bar")

    def test_three_token_leg_segment(self):
        assert pf.map_flybody_bname_to_flygym_bname("coxa_T1_left") == "lf_coxa"
        assert pf.map_flybody_bname_to_flygym_bname("tibia_T2_right") == "rm_tibia"
        assert pf.map_flybody_bname_to_flygym_bname("foo_T3_left") == "lh_foo"

    def test_three_token_tarsus_and_claw_aliases(self):
        assert pf.map_flybody_bname_to_flygym_bname("tarsus_T1_left") == "lf_tarsus1"
        assert pf.map_flybody_bname_to_flygym_bname("claw_T1_left") == "lf_tarsus5"

    def test_femur_is_rewritten_to_trochanterfemur(self):
        assert (
            pf.map_flybody_bname_to_flygym_bname("femur_T1_left")
            == "lf_trochanterfemur"
        )

    def test_three_token_invalid_leg_raises(self):
        with pytest.raises(ValueError):
            pf.map_flybody_bname_to_flygym_bname("coxa_X9_left")

    def test_four_token_claw_alias(self):
        assert (
            pf.map_flybody_bname_to_flygym_bname("tarsus_claw_T1_left") == "lf_tarsus5"
        )

    def test_four_token_numbered_segment(self):
        assert pf.map_flybody_bname_to_flygym_bname("seg_T1_2_left") == "lf_seg2"

    def test_four_token_invalid_raises(self):
        with pytest.raises(ValueError):
            pf.map_flybody_bname_to_flygym_bname("a_b_c_d")


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


class TestSplitWhitespaceToList:
    def test_multi_token(self):
        assert pf._split_whitespace_to_list("1 2 3") == ["1", "2", "3"]

    def test_single_token_returned_as_string(self):
        assert pf._split_whitespace_to_list("foo") == "foo"

    def test_empty_and_whitespace_only(self):
        assert pf._split_whitespace_to_list("") == []
        assert pf._split_whitespace_to_list("   ") == []

    def test_non_string_passthrough(self):
        assert pf._split_whitespace_to_list(5) == 5


class TestFirstOrValue:
    def test_list_returns_first(self):
        assert pf._first_or_value(["a", "b"]) == "a"

    def test_empty_list_returns_none(self):
        assert pf._first_or_value([]) is None

    def test_scalar_passthrough(self):
        assert pf._first_or_value("x") == "x"
        assert pf._first_or_value(7) == 7


class TestScaleNumericValue:
    def test_integer_result_is_stringified_int(self):
        assert pf._scale_numeric_value(2, 10) == "20"

    def test_string_numeric_input(self):
        assert pf._scale_numeric_value("3", 10) == "30"

    def test_non_integer_result(self):
        assert pf._scale_numeric_value(0.25, 10) == "2.5"

    def test_non_numeric_passthrough(self):
        assert pf._scale_numeric_value("abc", 10) == "abc"

    def test_none_returns_none(self):
        assert pf._scale_numeric_value(None, 10) is None


class TestParseAndScaleAttr:
    def test_scaled_list_attr(self):
        assert pf._parse_and_scale_attr("pos", "1 2 3") == ["10", "20", "30"]

    def test_scaled_scalar_attr(self):
        assert pf._parse_and_scale_attr("size", "0.5") == "5"

    def test_density_uses_fractional_scale(self):
        assert pf._parse_and_scale_attr("density", "1000") == "1"

    def test_unscaled_attr_passthrough(self):
        assert pf._parse_and_scale_attr("quat", "1 0 0 0") == ["1", "0", "0", "0"]
        assert pf._parse_and_scale_attr("name", "foo") == "foo"


class TestIsExcludedGeom:
    @pytest.mark.parametrize(
        "name", ["thorax_collision", "wing_fluid", "abdomen_inertial"]
    )
    def test_excluded(self, name):
        assert pf._is_excluded_geom(name) is True

    def test_not_excluded(self):
        assert pf._is_excluded_geom("thorax_black") is False


class TestMeshNameHelpers:
    def test_extract_base_and_suffix(self):
        assert pf._extract_mesh_base_and_suffix("thorax_black") == ("thorax", "black")
        assert pf._extract_mesh_base_and_suffix("wing_left_membrane") == (
            "wing_left",
            "membrane",
        )

    def test_extract_no_suffix_defaults_to_body(self):
        assert pf._extract_mesh_base_and_suffix("thorax") == ("thorax", "body")

    def test_translate_mesh_name(self):
        assert pf.translate_mesh_name("thorax_black") == "c_thorax_black"
        assert pf.translate_mesh_name("coxa_T1_left_black") == "lf_coxa_black"

    def test_translate_mesh_name_no_suffix(self):
        assert pf.translate_mesh_name("thorax") == "c_thorax_body"

    def test_collect_segment_suffixes_groups_and_sorts(self):
        result = pf.collect_segment_suffixes(["thorax_red", "thorax_black"])
        assert result == {"c_thorax": ["black", "red"]}


# ---------------------------------------------------------------------------
# Actuator-default resolution helpers
# ---------------------------------------------------------------------------


class TestActuatorHelpers:
    def test_is_ignored_actuator_class(self):
        assert pf._is_ignored_actuator_class("leg_collision") is True
        assert pf._is_ignored_actuator_class("tarsus_adhesion") is True
        assert pf._is_ignored_actuator_class("general") is False

    def test_clean_actuator_tag_config_drops_ctrlrange_for_general(self):
        cfg = {"ctrlrange": ["-1", "1"], "gainprm": ["1"]}
        cleaned = pf._clean_actuator_tag_config("general", cfg, ignore_ctrlrange=True)
        assert cleaned == {"gainprm": ["1"]}
        # original is not mutated
        assert "ctrlrange" in cfg

    def test_clean_actuator_tag_config_keeps_ctrlrange_when_not_ignored(self):
        cfg = {"ctrlrange": ["-1", "1"], "gainprm": ["1"]}
        assert (
            pf._clean_actuator_tag_config("general", cfg, ignore_ctrlrange=False) == cfg
        )

    def test_clean_actuator_tag_config_keeps_ctrlrange_for_non_general(self):
        cfg = {"ctrlrange": ["-1", "1"]}
        assert pf._clean_actuator_tag_config("motor", cfg, ignore_ctrlrange=True) == cfg

    def test_scale_actuator_tag_config(self):
        scaled = pf._scale_actuator_tag_config({"gainprm": "1", "foo": "bar"})
        assert scaled == {"gainprm": "10", "foo": "bar"}

    def test_has_meaningful_local_actuation(self):
        hierarchy = {
            "actu": {"parent": None, "local": {"general": {"gainprm": ["1"]}}},
            "ctrl_only": {
                "parent": None,
                "local": {"general": {"ctrlrange": ["-1", "1"]}},
            },
            "no_actu": {"parent": None, "local": {"joint": {"stiffness": "1"}}},
        }
        tags = {"general", "motor"}
        assert pf._has_meaningful_local_actuation("actu", hierarchy, tags, True) is True
        # ctrlrange-only general becomes empty under ignore_ctrlrange
        assert (
            pf._has_meaningful_local_actuation("ctrl_only", hierarchy, tags, True)
            is False
        )
        assert (
            pf._has_meaningful_local_actuation("no_actu", hierarchy, tags, True)
            is False
        )
        assert (
            pf._has_meaningful_local_actuation("missing", hierarchy, tags, True)
            is False
        )

    def test_resolve_representative_class_walks_to_ancestor(self):
        hierarchy = {
            "leaf": {"parent": "base", "local": {}},
            "base": {"parent": None, "local": {"general": {"gainprm": ["1"]}}},
        }
        tags = {"general"}
        assert pf._resolve_representative_class("leaf", hierarchy, tags, True) == "base"

    def test_resolve_representative_class_ignored_returns_none(self):
        hierarchy = {"leg_collision": {"parent": None, "local": {}}}
        assert (
            pf._resolve_representative_class(
                "leg_collision", hierarchy, {"general"}, True
            )
            is None
        )

    def test_resolve_representative_class_no_actuation_returns_none(self):
        hierarchy = {"plain": {"parent": None, "local": {"joint": {"stiffness": "1"}}}}
        assert (
            pf._resolve_representative_class("plain", hierarchy, {"general"}, True)
            is None
        )

    def test_merge_equivalent_actuator_groups_merges_identical(self):
        groups = {
            "a": {"general": {"gainprm": ["1"]}, "apply_to": ["j1"]},
            "b": {"general": {"gainprm": ["1"]}, "apply_to": ["j2"]},
            "c": {"general": {"gainprm": ["2"]}, "apply_to": ["j3"]},
        }
        merged = pf._merge_equivalent_actuator_groups(groups)
        # a and b share a signature and collapse to "a-b"; c stays separate.
        assert "a-b" in merged
        assert "c" in merged
        assert merged["a-b"]["apply_to"] == ["j1", "j2"]
        assert merged["c"]["apply_to"] == "j3"

    def test_group_joint_params_for_yaml_groups_shared_and_splits_ranges(self):
        all_joints = {
            "j1": {"stiffness": "1", "range": ["0", "1"]},
            "j2": {"stiffness": "1", "range": ["2", "3"]},
            "j3": {"stiffness": "9"},
        }
        out = pf._group_joint_params_for_yaml(all_joints)
        # j1 and j2 share shared-params {stiffness: 1}; j3 differs.
        groups = out["params"]
        assert len(groups) == 2
        apply_targets = [g["apply_to"] for g in groups.values()]
        assert ["j1", "j2"] in apply_targets
        assert "j3" in apply_targets
        # ranges captured per-joint, only where present
        assert out["ranges"] == {
            "j1": {"range": ["0", "1"]},
            "j2": {"range": ["2", "3"]},
        }


# ---------------------------------------------------------------------------
# get_flygym_jointname (operates on ET elements)
# ---------------------------------------------------------------------------


class TestGetFlygymJointName:
    def _bodies(self):
        parent = ET.fromstring('<body name="thorax"/>')
        child = ET.fromstring('<body name="coxa_T1_left"/>')
        return parent, child

    def test_dof_from_joint_name(self):
        parent, child = self._bodies()
        joint = ET.fromstring('<joint name="coxa_T1_left_pitch"/>')
        assert pf.get_flygym_jointname(parent, child, joint) == "c_thorax-lf_coxa-pitch"

    def test_dof_from_joint_class_alias(self):
        parent, child = self._bodies()
        joint = ET.fromstring('<joint name="j" class="twist"/>')
        # twist -> roll per dof_mapping
        assert pf.get_flygym_jointname(parent, child, joint) == "c_thorax-lf_coxa-roll"

    def test_axis_fallback_to_pitch(self):
        parent, child = self._bodies()
        joint = ET.fromstring('<joint name="j" axis="1 0 0"/>')
        assert pf.get_flygym_jointname(parent, child, joint) == "c_thorax-lf_coxa-pitch"

    def test_wrong_axis_raises(self):
        parent, child = self._bodies()
        joint = ET.fromstring('<joint name="j" axis="0 1 0"/>')
        with pytest.raises(AssertionError):
            pf.get_flygym_jointname(parent, child, joint)

    def test_missing_axis_warns_and_defaults(self):
        parent, child = self._bodies()
        joint = ET.fromstring('<joint name="j"/>')
        with pytest.warns(UserWarning):
            name = pf.get_flygym_jointname(parent, child, joint)
        assert name == "c_thorax-lf_coxa-pitch"


# ---------------------------------------------------------------------------
# XML-driven parsing: synthetic MuJoCo model
# ---------------------------------------------------------------------------


MODEL_XML = """\
<mujoco model="test">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.0001" gravity="0 0 -9.81"/>
  <size njmax="500" nconmax="100"/>
  <default>
    <geom density="1000"/>
    <default class="body">
      <joint stiffness="10" damping="1"/>
      <default class="leg">
        <joint stiffness="20"/>
      </default>
    </default>
    <default class="actu">
      <general gainprm="1 0 0" ctrlrange="-1 1"/>
    </default>
  </default>
  <asset>
    <material name="black" rgba="0 0 0 1"/>
  </asset>
  <worldbody>
    <body name="thorax" pos="0 0 0" childclass="body">
      <geom name="thorax_black" mesh="thorax_black"/>
      <body name="coxa_T1_left" pos="0.1 0 0" childclass="leg">
        <joint name="coxa_T1_left_pitch" class="actu" range="-1 1" springref="0.5"/>
        <geom name="coxa_black" mesh="coxa_T1_left_black"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def model_root():
    return ET.fromstring(MODEL_XML)


@pytest.fixture
def xml_path(tmp_path):
    p = tmp_path / "model.xml"
    p.write_text(MODEL_XML)
    return p


class TestBuildEffectiveDefaultLookup:
    def test_root_and_class_inheritance(self, model_root):
        lookup = pf.build_effective_default_lookup(model_root)
        assert lookup["__root__"]["geom"]["density"] == "1000"
        assert lookup["body"]["joint"]["stiffness"] == "10"
        # leg overrides stiffness but inherits damping from body
        assert lookup["leg"]["joint"]["stiffness"] == "20"
        assert lookup["leg"]["joint"]["damping"] == "1"
        assert lookup["actu"]["general"]["gainprm"] == ["1", "0", "0"]

    def test_missing_default_section_raises(self):
        root = ET.fromstring("<mujoco><worldbody/></mujoco>")
        with pytest.raises(ValueError):
            pf.build_effective_default_lookup(root)


class TestCollectDefaultClassHierarchy:
    def test_parent_links_and_local_params(self, model_root):
        hierarchy = pf._collect_default_class_hierarchy(model_root)
        assert hierarchy["body"]["parent"] is None
        assert hierarchy["leg"]["parent"] == "body"
        # local params are non-inherited: leg only declares stiffness locally
        assert hierarchy["leg"]["local"]["joint"] == {"stiffness": "20"}


class TestAddClassParams:
    def test_accumulates_tag_params(self):
        lookup = {"leg": {"joint": {"stiffness": "20"}}}
        acc = {}
        pf.add_class_params(lookup, "joint", "leg", acc)
        assert acc == {"stiffness": "20"}

    def test_unknown_class_raises(self):
        with pytest.raises(ValueError):
            pf.add_class_params({}, "joint", "missing", {})


class TestRecursiveAccumulationJointParams:
    def test_collects_joint_with_inherited_and_explicit_params(self, model_root):
        lookup = pf.build_effective_default_lookup(model_root)
        worldbody = model_root.find("worldbody")
        joints = pf.recursive_accumulation_joint_params(worldbody, {}, lookup)
        key = "c_thorax-lf_coxa-pitch"
        assert key in joints
        # stiffness from the "leg" childclass, range from the explicit joint attr
        assert joints[key]["stiffness"] == "20"
        assert joints[key]["range"] == ["-1", "1"]
        assert joints[key]["springref"] == "0.5"


class TestCollectClassApplyTargets:
    def test_maps_joint_class_to_flygym_joint_name(self, model_root):
        targets = pf._collect_class_apply_targets(model_root.find("worldbody"))
        assert targets["actu"] == {"c_thorax-lf_coxa-pitch"}


# ---------------------------------------------------------------------------
# End-to-end parse_* writers
# ---------------------------------------------------------------------------


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


class TestParseWriters:
    def test_parse_globals(self, xml_path, tmp_path):
        out = tmp_path / "globals.yaml"
        pf.parse_globals(xml_path, out)
        data = _load_yaml(out)
        assert data["option"]["timestep"] == "0.0001"
        assert data["compiler"]["fusestatic"] == "true"
        assert data["statistic"] == {"extent": "5"}
        assert "headlight" in data["visual"]

    def test_parse_visuals(self, xml_path, tmp_path):
        out = tmp_path / "visuals.yaml"
        pf.parse_visuals(xml_path, out)
        data = _load_yaml(out)
        assert data["black"]["apply_to"] == "*_black"
        assert data["black"]["material"]["rgba"] == ["0", "0", "0", "1"]

    def test_parse_joints(self, xml_path, tmp_path):
        joints_out = tmp_path / "joints.yaml"
        pose_out = tmp_path / "pose.yaml"
        pf.parse_joints(xml_path, joints_out, pose_out, kin_order="yaw_roll_pitch")
        joints = _load_yaml(joints_out)
        pose = _load_yaml(pose_out)
        assert "params" in joints and "ranges" in joints
        assert joints["ranges"]["c_thorax-lf_coxa-pitch"]["range"] == ["-1", "1"]
        assert pose["axis_order"] == ["yaw", "roll", "pitch"]
        assert pose["joint_angles"]["c_thorax-lf_coxa-pitch"] == 0.5

    def test_parse_actuators(self, xml_path, tmp_path):
        out = tmp_path / "actuators.yaml"
        pf.parse_actuators(xml_path, out)
        data = _load_yaml(out)
        # single actuator class "actu" with a general tag applied to the joint
        assert len(data) == 1
        group = next(iter(data.values()))
        assert "general" in group
        assert group["apply_to"] == "c_thorax-lf_coxa-pitch"
        # ctrlrange dropped by default (ignore_ctrlrange=True)
        assert "ctrlrange" not in group["general"]

    def test_parse_xml_to_rig(self, xml_path, tmp_path):
        out = tmp_path / "rigging.yaml"
        pf.parse_xml_to_rig(xml_path, out)
        rig = _load_yaml(out)
        assert rig["c_thorax"]["flybody_name"] == "thorax"
        assert "c_thorax_black" in rig["c_thorax"]["geoms"]
        assert rig["lf_coxa"]["flybody_name"] == "coxa_T1_left"
        # density default propagated and scaled (1000 * 1e-3 -> "1")
        assert rig["c_thorax"]["geoms"]["c_thorax_black"]["density"] == "1"
        # sibling suffix file is written next to the rig
        assert (out.with_name("all_geom_suffixes.yaml")).exists()
