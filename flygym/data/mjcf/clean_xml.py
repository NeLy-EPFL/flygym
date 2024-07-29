from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from lxml import etree
from xml.dom import minidom
import os

xml_path = Path(
    "/Users/stimpfli/Desktop/flygym_other/flygym/data/mjcf/groundwalking_nmf_mjcf_nofloor_230518__bendTarsus_scaled.xml"
)
# for seqikpy should be yaw pitch roll for df3d data should be roll yaw pitch
kin_chain_order = ["yaw", "pitch", "roll"]
only_remove_visual_geoms = False


def get_symmetrical_bodies(xml):
    symmetrical_bodies = []
    centered_bodies = []
    for a in xml.findall("//body"):
        name = a.attrib.get("name")
        if name.startswith("R"):
            if xml.find(f"//body[@name='L{name[1:]}']") is not None:
                symmetrical_bodies.append(name[1:])
            else:
                centered_bodies.append(name)
        else:
            if (
                xml.find(f"//body[@name='R{name[1:]}']") is None
                and "roll" not in name
                and "yaw" not in name
            ):
                centered_bodies.append(name)

    return symmetrical_bodies, centered_bodies


def check_is_dummy(node, eps=1e-5):
    checks = []
    for attrib in node.attrib.keys():
        vals = np.array(node.attrib[attrib].split(" ")).astype(float)
        if np.all(np.abs(vals) < eps):
            checks.append(True)
        else:
            checks.append(False)
    return np.all(checks)


def check_is_at_origin(node, eps=1e-5):
    node_pos = np.array(node.attrib["pos"].split(" ")).astype(float)
    if np.all(np.abs(node_pos) < eps):
        return True
    else:
        return False


def set_mirrored_meshes(xml, symmetrical_bodies):
    # Use meshes from the right side for the left side by mirroring them
    # This is done by using a negative scaling factor on the y axis

    mesh_parent = xml.findall(".//mesh")[0].getparent()

    for mesh_name_template in symmetrical_bodies:
        R_mesh_name = "mesh_R" + mesh_name_template
        L_mesh_name = "mesh_L" + mesh_name_template
        R_mesh = xml.find(f"//mesh[@name='{R_mesh_name}']")
        L_mesh = xml.find(f"//mesh[@name='{L_mesh_name}']")

        if L_mesh is not None and R_mesh is not None:
            R_mesh_rot = deepcopy(R_mesh)
            L_mesh_scaling = np.array(
                [float(a) for a in L_mesh.get("scale").split()], dtype=np.int32
            )
            L_mesh_scaling[1] *= -1
            R_mesh_rot.set(
                "scale", " ".join([str(a) for a in L_mesh_scaling])
            )  # flip around the plane x=0
            R_mesh_rot.set("name", L_mesh_name)
            mesh_parent.remove(L_mesh)
            mesh_parent.append(R_mesh_rot)
        else:
            print(f"Meshes {R_mesh_name} or {L_mesh_name} not found")
    return xml


def make_symmetric(xml, symmetrical_bodies, centered_bodies):
    # Make the xml symmetrical by setting the origin of every body symmetric to the origin
    # This is done for now by taking the mean of the two possible non symmetrical position

    symmetry_vect = np.array([1, -1, 1])

    for sym_name_base in symmetrical_bodies:
        body_R = xml.find(f"//body[@name='R{sym_name_base}']")
        body_L = xml.find(f"//body[@name='L{sym_name_base}']")

        pos_R = np.array([np.longdouble(a) for a in body_R.attrib.get("pos").split()])
        pos_L = np.array([np.longdouble(a) for a in body_L.attrib.get("pos").split()])
        sym_pos_L = deepcopy(pos_L) * symmetry_vect
        sym_pos_R = deepcopy(pos_R) * symmetry_vect

        new_pos_L = (sym_pos_R + pos_L) / 2
        new_pos_R = (sym_pos_L + pos_R) / 2

        assert np.all(
            new_pos_L * symmetry_vect == new_pos_R
        ), f"Symmetry did not work for {sym_name_base}"

        body_R.set("pos", " ".join([str(a) for a in new_pos_R]))
        body_L.set("pos", " ".join([str(a) for a in new_pos_L]))

    for body_name in centered_bodies:
        body = xml.find(f"//body[@name='{body_name}']")
        body_pos = np.array([float(a) for a in body.attrib.get("pos").split()])
        body_pos[1] = 0.0
        body.set("pos", " ".join([str(a) for a in body_pos]))

    return xml


def clean_geoms(xml):
    # clean the geometries by:
    # - removing visual geoms
    # - removing the suffix of the geoms
    # - set their pos to 0 0 0
    # - set the contype and conaffinity to 0

    final_geom_names = []
    for geom in xml.findall("//geom"):
        geom_name = geom.get("name")
        final_geom_name = geom_name.split("_")[0]
        if final_geom_name not in final_geom_names:
            final_geom_names.append(final_geom_name)

    for final_geom_name in final_geom_names:
        visual_geom = xml.find(f".//geom[@name='{final_geom_name}_visual']")
        collision_geom = xml.find(f".//geom[@name='{final_geom_name}_collision']")
        final_mesh_name = ""
        if collision_geom is not None:
            final_mesh_name = collision_geom.get("mesh").replace("_collision", "")
        else:
            final_mesh_name = visual_geom.get("mesh").replace("_visual", "")

        assert (
            visual_geom is not None
        ), f"Visual geom {final_geom_name}_visual not found the file does not follow the basic naming convention"

        if (collision_geom is not None) and (visual_geom is not None):
            # This geom has both a visual and a collision geom: we should remove the visual geom and the mesh referencing it and rename the collision geom
            visual_mesh_name = visual_geom.get("mesh")
            visual_mesh = xml.find(f".//mesh[@name='{visual_mesh_name}']")
            collision_mesh_name = collision_geom.get("mesh")
            collision_mesh = xml.find(f".//mesh[@name='{collision_mesh_name}']")

            # remove the visual geom and the mesh
            visual_geom.getparent().remove(visual_geom)
            if visual_mesh is not None:
                visual_mesh.getparent().remove(visual_mesh)
            else:
                geom_name = visual_geom.get("name")
                print(f"Mesh {visual_mesh_name} for geom {geom_name} not found")

            # set to the right values
            if collision_mesh is not None:
                collision_mesh.set("name", final_mesh_name)
            else:
                print(
                    f"Mesh {collision_mesh_name} for geom {collision_geom.get('name')} not found"
                )
            collision_geom.set("name", final_geom_name)
            collision_geom.set("mesh", final_mesh_name)
            collision_geom.set("pos", "0 0 0")
            collision_geom.set("contype", "0")
            collision_geom.set("conaffinity", "0")

        elif collision_geom is None:
            # This geom has no collision geom: just change its name and the name of the mesh referencing it
            mesh_name = visual_geom.get("mesh")
            mesh = xml.find(f".//mesh[@name='{mesh_name}']")
            # set to the right values
            if mesh is not None:
                mesh.set("name", final_mesh_name)
            else:
                print(f"Mesh {mesh_name} for geom {visual_geom.get('name')} not found")
            visual_geom.set("name", final_geom_name)
            visual_geom.set("mesh", final_mesh_name)
            visual_geom.set("pos", "0 0 0")
            visual_geom.set("contype", "0")
            visual_geom.set("conaffinity", "0")

    return xml


def get_all_multidof_joints(xml):
    root = xml.getroot()
    multi_dof_joints = []
    multi_dof_joint = []
    n_dofs = 0
    for line in root.iter():
        if line.tag == "body":
            body_name = line.attrib.get("name")
            root_body_name = body_name.replace("_roll", "").replace("_yaw", "")

            for child in list(line):
                child_name = child.get("name")
                has_dof_child = False
                if (
                    (child_name is not None)
                    and (root_body_name in child_name)
                    and (child.tag == "body")
                ):
                    multi_dof_joint.append(deepcopy(body_name))
                    n_dofs += 1
                    assert not has_dof_child, f"{body_name} {child_name}"
                    has_dof_child = True
            if n_dofs > 0 and not has_dof_child:
                n_dofs = 0
                multi_dof_joint.append(body_name)
                multi_dof_joints.append(multi_dof_joint)
                multi_dof_joint = []

    return multi_dof_joints


def get_body_min_descendant(xml, names):
    n_descendants = []
    for name in names:
        node = xml.find(f"//body[@name='{name}']")
        n_descendants.append(len(list(node.iterdescendants())))
    return min(n_descendants)


def remove_dummy_bodies(xml, kin_chain_order=["yaw", "pitch", "roll"]):
    # remove dummy bodies that form the multi dof joints

    multi_dof_joint = get_all_multidof_joints(xml)
    # start at the bottom of the chain for safety not sure this is necessary
    multi_dof_joint = sorted(
        multi_dof_joint, key=lambda x: get_body_min_descendant(xml, x)
    )

    for body_names in multi_dof_joint:
        # find the pitch degree of freedom that has no suffix
        base_body_name = min(body_names, key=len)
        old_base_body = xml.find(f"//body[@name='{body_names[0]}']")
        base_parent = old_base_body.getparent()
        base_body = etree.Element("body")

        for attrib in old_base_body.attrib.keys():
            if not attrib == "name":
                base_body.set(attrib, old_base_body.get(attrib))
            else:
                base_body.set("name", base_body_name)

        joints = []
        joints_names = []
        all_children = []
        n_dofs = len(body_names)

        for b, body_name in enumerate(body_names):
            body = xml.find(f".//body[@name='{body_name}']")

            # handle the joints
            joint = body.find("joint")
            if joint is not None:
                joints.append(joint)
                joints_names.append(joint.get("name"))
            else:
                print("No joint found for body", body_name)

            # handle the inertia
            inertial = body.find("inertial")
            if inertial is not None and not check_is_dummy(inertial):
                base_body.append(inertial)

            # handle the geoms
            geoms = body.findall("geom")
            for geom in geoms:
                base_body.append(geom)

            # if is the last joint
            if b >= (n_dofs - 1):
                # copy rest of kinematic chain
                for child in list(body):
                    if child.tag == "body":
                        all_children.append(child)
                    elif child.tag not in ["joint", "inertial", "geom"]:
                        print(f"Child {child.tag} not handled for body {body_name}")

        # reorder the joints and add them
        new_joint_order = [None, None, None]
        for j, joints_name in enumerate(joints_names):
            dof = joints_name.split("_")[-1]
            if dof not in ["yaw", "roll"]:
                dof = "pitch"
            new_joint_order[kin_chain_order.index(dof)] = j
        # remove the None values
        new_joint_order = [k for k in new_joint_order if k is not None]
        reordered_joints = [joints[k] for k in new_joint_order]
        for joint in reordered_joints:
            base_body.append(joint)

        # add the children
        for child in all_children:
            base_body.append(child)

        base_parent.append(base_body)
        # remove the old parent body
        base_parent.remove(old_base_body)
    return xml


def remove_inertials(xml):
    for inertial in xml.findall("//inertial"):
        body = inertial.getparent()
        geom = body.find("geom")
        if geom is not None:
            geom.set("mass", inertial.get("mass"))
        else:
            print(f"No geom found for body {body.get('name')}")
        body.remove(inertial)
    return xml


def save_xml(xml, path):
    # xml.write(path)

    # reparse the xml with nicely aligned children
    rough_string = etree.tostring(xml).decode("utf-8")
    reparsed = minidom.parseString(rough_string)
    smooth_string = reparsed.toprettyxml()
    smooth_string = os.linesep.join(
        [s for s in smooth_string.splitlines() if s.strip()]
    )
    with open(path, "w") as f:
        f.write(smooth_string)


def clean_xml(
    xml_path, only_remove_visual_geoms=False, kin_chain_order=["yaw", "pitch", "roll"]
):
    tree = etree.parse(xml_path)
    # check all the joints
    symmetrical_bodies, centered_bodies = get_symmetrical_bodies(tree)

    tree = clean_geoms(tree)
    if only_remove_visual_geoms:
        filename = f"{xml_path.stem}_clean_geoms.xml"
        new_xml_path = xml_path.parent / filename
        save_xml(tree, new_xml_path)
        return None

    else:
        tree = remove_dummy_bodies(tree)
        # Remove the dummy bodies names from the symmetrical bodies
        symmetrical_bodies = [
            body
            for body in symmetrical_bodies
            if "roll" not in body and "yaw" not in body
        ]

        tree = remove_inertials(tree)

        tree = make_symmetric(tree, symmetrical_bodies, centered_bodies)

        tree = set_mirrored_meshes(tree, symmetrical_bodies)

        kin_order_short = "".join([k[0] for k in kin_chain_order])
        filename = f"nmf_mjcf_240305_sym_kinorder_{kin_order_short}.xml"
        new_xml_path = xml_path.parent / filename
        save_xml(tree, new_xml_path)
        return None


xml = clean_xml(xml_path, only_remove_visual_geoms, kin_chain_order)
