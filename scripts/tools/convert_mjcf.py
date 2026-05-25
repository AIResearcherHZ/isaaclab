# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a MJCF into USD format.

MuJoCo XML Format (MJCF) is an XML file format used in MuJoCo to describe all elements of a robot.
For more information, see: http://www.mujoco.org/book/XMLreference.html

This script uses the MJCF importer extension from Isaac Sim (``isaacsim.asset.importer.mjcf``) to convert
a MJCF asset into USD format. It is designed as a convenience script for command-line use. For more information
on the MJCF importer, see the documentation for the extension:
https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/ext_isaacsim_asset_importer_mjcf.html


positional arguments:
  input               The path to the input URDF file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                Show this help message and exit
  --fix-base                Fix the base to where it is imported. (default: False)
  --import-sites            Import sites by parse <site> tag. (default: True)
  --make-instanceable       Make the asset instanceable for efficient cloning. (default: False)

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Utility to convert a MJCF into USD format.")
parser.add_argument("input", type=str, help="The path to the input MJCF file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")
parser.add_argument("--fix-base", action="store_true", default=False, help="Fix the base to where it is imported.")
parser.add_argument(
    "--import-sites", action="store_true", default=False, help="Import sites by parsing the <site> tag."
)
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=False,
    help="Make the asset instanceable for efficient cloning.",
)
parser.add_argument(
    "--no-flatten",
    action="store_true",
    default=False,
    help=(
        "Skip flattening and instead write a thin wrapper .usd at the output path that references the importer's"
        " package directory. Default is to produce a single self-contained .usd at the output path."
    ),
)
parser.add_argument(
    "--remove-package",
    action="store_true",
    default=False,
    help=(
        "After flattening, delete the intermediate <out_dir>/<robot_name>/ package directory. Ignored when"
        " --no-flatten is set."
    ),
)
parser.add_argument(
    "--no-flat-hierarchy",
    action="store_true",
    default=False,
    help=(
        "Skip the post-flatten step that reparents every link to a direct child of the root prim and"
        " moves joints under <root>/joints."
    ),
)
parser.add_argument(
    "--keep-empty-scopes",
    action="store_true",
    default=False,
    help="After flattening the hierarchy, keep intermediate scopes (Geometry/Physics/...) even if empty.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
import shutil

import carb
import omni.kit.app
import omni.usd  # noqa: F401 — bind `omni` at module scope so the GUI branch below doesn't trip the local-scope rule

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import check_file_path

# isaacsim 5.1 ships the new MJCF importer 3.2.0 with a pure-Python API. The old
# MjcfConverter under isaaclab.sim.converters relies on the C++ Kit command
# "MJCFCreateImportConfig" which is no longer registered (raises AttributeError
# on a None config). We use the new API directly here.
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig


# ---------------------------------------------------------------------------
# Post-import flatten: bring every rigid-body link directly under the root prim
# and move every physics joint under <root>/joints, so prim_paths like
# {ENV_REGEX_NS}/Robot/<link> resolve under Isaac Lab's non-recursive matcher.
#
# MJCF importer 3.2.0 emits the kinematic tree as a nested chain under
# /<Robot>/Geometry/world/<base>/<child>/.../<leaf>. Without this pass,
# FrameTransformerCfg / find_matching_prims won't see the links.
# ---------------------------------------------------------------------------
def _anchor_chain_root_to_world(stage, root_path, joints_scope_path) -> bool:
    """Pin the kinematic chain's root link to the simulation world frame.

    Detects the chain root — a link that is body0 of some joint but never body1 of a
    non-world joint — and, if it isn't already rigidly connected to a world-pinned
    link, adds a FixedJoint that anchors it. Re-uses an existing dangling world-pinned
    link as the anchor when present.
    """
    from pxr import Usd, UsdPhysics

    root_prim = stage.GetPrimAtPath(root_path)
    joints_scope = stage.GetPrimAtPath(joints_scope_path)
    if not root_prim or not joints_scope:
        return False

    link_paths = {
        p.GetPath() for p in Usd.PrimRange(root_prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)
    }

    parents, children, world_anchored, fixed_pairs = set(), set(), set(), []
    for jp in joints_scope.GetChildren():
        joint = UsdPhysics.Joint(jp)
        if not joint:
            continue
        b0 = list(joint.GetBody0Rel().GetTargets())
        b1 = list(joint.GetBody1Rel().GetTargets())
        is_fixed = jp.IsA(UsdPhysics.FixedJoint)
        if is_fixed and ((not b0 and len(b1) == 1) or (not b1 and len(b0) == 1)):
            anchor = (b1 or b0)[0]
            if anchor in link_paths:
                world_anchored.add(anchor)
            continue
        if len(b1) == 1 and b1[0] in link_paths:
            children.add(b1[0])
        if len(b0) == 1 and b0[0] in link_paths:
            parents.add(b0[0])
        if is_fixed and len(b0) == 1 and len(b1) == 1:
            fixed_pairs.append((b0[0], b1[0]))

    chain_roots = parents - children
    if not chain_roots:
        return False

    adj = {}
    for a, b in fixed_pairs:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def anchored_to_world(start):
        if start in world_anchored:
            return True
        seen, stack = {start}, [start]
        while stack:
            x = stack.pop()
            for y in adj.get(x, ()):
                if y in world_anchored:
                    return True
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return False

    added = False
    for chain_root in chain_roots:
        if anchored_to_world(chain_root):
            continue
        new_joint_path = joints_scope_path.AppendChild(f"_fix_base_{chain_root.name}")
        if stage.GetPrimAtPath(new_joint_path):
            continue
        new_joint = UsdPhysics.FixedJoint.Define(stage, new_joint_path)
        if world_anchored:
            anchor = sorted(world_anchored, key=lambda p: p.pathString)[0]
            new_joint.CreateBody0Rel().SetTargets([anchor])
            new_joint.CreateBody1Rel().SetTargets([chain_root])
        else:
            new_joint.CreateBody0Rel().SetTargets([])
            new_joint.CreateBody1Rel().SetTargets([chain_root])
        added = True
    return added


def _flatten_hierarchy(usd_path: str, *, drop_empty_scopes: bool = True, fix_base: bool = False) -> dict:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    joint_schemas = (
        UsdPhysics.Joint,
        UsdPhysics.RevoluteJoint,
        UsdPhysics.PrismaticJoint,
        UsdPhysics.FixedJoint,
        UsdPhysics.SphericalJoint,
        UsdPhysics.DistanceJoint,
    )

    def _is_joint(prim):
        return any(prim.IsA(s) for s in joint_schemas)

    def _set_local_to_matrix(prim, mat):
        xf = UsdGeom.Xformable(prim)
        if not xf:
            return
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(mat)

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open USD: {usd_path}")
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        raise RuntimeError(f"USD has no defaultPrim: {usd_path}")
    root_path = root_prim.GetPath()

    links, joints = [], []
    for prim in Usd.PrimRange(root_prim):
        if prim == root_prim:
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            links.append(prim)
        if _is_joint(prim):
            joints.append(prim)

    if not links and not joints:
        return {"links": 0, "joints": 0, "deleted_scopes": 0}

    xf_cache = UsdGeom.XformCache()
    link_world_xf = {p.GetName(): xf_cache.GetLocalToWorldTransform(p) for p in links}
    link_names = {p.GetName() for p in links}
    joint_names = [p.GetName() for p in joints]

    joints_scope_path = root_path.AppendChild("joints")
    already_flat = all(p.GetPath().GetParentPath() == root_path for p in links) and all(
        p.GetPath().GetParentPath() == joints_scope_path for p in joints
    )

    if not already_flat:
        UsdGeom.Scope.Define(stage, joints_scope_path)
        edits = Sdf.BatchNamespaceEdit()
        # Reparent deepest-first so a link's children detach before the link itself.
        for prim in sorted(links, key=lambda p: -len(p.GetPath().pathString.split("/"))):
            old = prim.GetPath()
            if old.GetParentPath() == root_path:
                continue
            if stage.GetPrimAtPath(root_path.AppendChild(old.name)):
                raise RuntimeError(f"Name collision at {root_path.AppendChild(old.name)} while reparenting {old}.")
            edits.Add(Sdf.NamespaceEdit.Reparent(old, root_path, -1))
        for prim in joints:
            old = prim.GetPath()
            if old.GetParentPath() == joints_scope_path:
                continue
            if stage.GetPrimAtPath(joints_scope_path.AppendChild(old.name)):
                raise RuntimeError(
                    f"Name collision at {joints_scope_path.AppendChild(old.name)} while reparenting {old}."
                )
            edits.Add(Sdf.NamespaceEdit.Reparent(old, joints_scope_path, -1))

        layer = stage.GetRootLayer()
        if not layer.Apply(edits):
            raise RuntimeError("BatchNamespaceEdit failed — likely a name collision at the destination.")

        # Joint body rels were nested paths; rewrite by basename.
        for joint_name in joint_names:
            jp = stage.GetPrimAtPath(joints_scope_path.AppendChild(joint_name))
            joint = UsdPhysics.Joint(jp) if jp and jp.IsValid() else None
            if not joint:
                continue
            for rel in (joint.GetBody0Rel(), joint.GetBody1Rel()):
                new_targets, changed = [], False
                for target in rel.GetTargets():
                    if target != root_path and target.name in link_names:
                        new_targets.append(root_path.AppendChild(target.name))
                        changed = True
                    else:
                        new_targets.append(target)
                if changed:
                    rel.SetTargets(new_targets)

        # New parent /Robot is identity; bake each link's world xform.
        for link_name, world_xf in link_world_xf.items():
            new_prim = stage.GetPrimAtPath(root_path.AppendChild(link_name))
            if not new_prim or not new_prim.IsValid():
                raise RuntimeError(f"Post-reparent: missing prim at {root_path.AppendChild(link_name)}")
            _set_local_to_matrix(new_prim, world_xf)

    # ArticulationRootAPI must live on the prim whose subtree contains the joints.
    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None
    for prim in stage.Traverse():
        if prim.GetPath() == root_path:
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if PhysxSchema and prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
                PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
    if not root_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)

    # Isaac Lab's find_global_fixed_joint_prim only picks up a fixed-base joint when
    # exactly one body rel is empty. Match that convention here.
    for joint_name in joint_names:
        jp = stage.GetPrimAtPath(joints_scope_path.AppendChild(joint_name))
        if not jp or not jp.IsValid() or not jp.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.Joint(jp)
        b0 = joint.GetBody0Rel().GetTargets()
        b1 = joint.GetBody1Rel().GetTargets()
        if len(b0) == 1 and b0[0] == root_path and len(b1) == 1:
            joint.GetBody0Rel().SetTargets([])

    for rel_name in ("isaac:physics:robotJoints", "isaac:physics:robotLinks"):
        rel = root_prim.GetRelationship(rel_name)
        if not rel:
            continue
        rel.SetTargets([
            root_path.AppendChild(t.name) if t.name in link_names
            else joints_scope_path.AppendChild(t.name) if t.name in joint_names
            else t
            for t in rel.GetTargets()
        ])

    deleted = 0
    if drop_empty_scopes:
        protected = {"joints"} | link_names
        for child in list(root_prim.GetChildren()):
            name = child.GetName()
            if name in protected:
                continue
            if name == "Materials" and child.GetChildren():
                continue
            keep = False
            for sub in Usd.PrimRange(child):
                if sub == child:
                    continue
                if sub.HasAPI(UsdPhysics.RigidBodyAPI) or _is_joint(sub):
                    keep = True
                    break
            if not keep:
                stage.RemovePrim(child.GetPath())
                deleted += 1

    if fix_base:
        _anchor_chain_root_to_world(stage, root_path, joints_scope_path)

    stage.GetRootLayer().Save()
    return {"links": len(links), "joints": len(joints), "deleted_scopes": deleted}


def main():
    mjcf_path = args_cli.input
    if not os.path.isabs(mjcf_path):
        mjcf_path = os.path.abspath(mjcf_path)
    if not check_file_path(mjcf_path):
        raise ValueError(f"Invalid file path: {mjcf_path}")
    # The 3.2.0 importer writes <out_dir>/<robot_name>/<robot_name>.usda; we additionally
    # produce a self-contained binary .usd at the user-requested dest_path.
    dest_path = args_cli.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)
    out_dir = os.path.dirname(dest_path)
    os.makedirs(out_dir, exist_ok=True)
    robot_name = os.path.splitext(os.path.basename(mjcf_path))[0]

    # CLI flags kept for backwards compat — 3.2.0 has no equivalent, must be applied post-hoc.
    # --fix-base is now applied via _anchor_chain_root_to_world in the flatten pass.
    unsupported = []
    if args_cli.import_sites:
        unsupported.append("--import-sites")
    if args_cli.make_instanceable:
        unsupported.append("--make-instanceable")
    if unsupported:
        print(
            f"[WARN] convert_mjcf.py: the following flags are ignored under MJCF importer 3.2.0 "
            f"({', '.join(unsupported)}); apply them as USD edits on the produced asset if needed."
        )

    print("-" * 80)
    print(f"Input MJCF file : {mjcf_path}")
    print(f"Output directory: {out_dir}")
    print(f"Robot name      : {robot_name}")
    print("-" * 80)

    cfg = MJCFImporterConfig()
    cfg.mjcf_path = mjcf_path
    cfg.usd_path = out_dir
    cfg.import_scene = True
    cfg.merge_mesh = False
    cfg.collision_from_visuals = False
    cfg.allow_self_collision = False
    importer = MJCFImporter(cfg)
    produced_usda = importer.import_mjcf()
    print(f"Generated USD package: {produced_usda}")

    # Flatten the importer's intermediate .usda in place so it matches the final .usd layout.
    if not args_cli.no_flat_hierarchy:
        try:
            pkg_summary = _flatten_hierarchy(
                produced_usda,
                drop_empty_scopes=not args_cli.keep_empty_scopes,
                fix_base=args_cli.fix_base,
            )
            print(
                f"Flat-hierarchy (package .usda): links={pkg_summary['links']}"
                f" joints={pkg_summary['joints']} scopes_deleted={pkg_summary['deleted_scopes']}"
            )
        except Exception as exc:
            print(f"[WARN] could not flatten package .usda: {exc}")

    # Default: flatten the package into a self-contained binary .usd at dest_path.
    # --no-flatten: write a thin wrapper .usd that references the package .usda.
    try:
        from pxr import Sdf, Usd

        if args_cli.no_flatten:
            wrapper_stage = (
                Usd.Stage.CreateNew(dest_path) if not os.path.exists(dest_path) else Usd.Stage.Open(dest_path)
            )
            if wrapper_stage is None:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                wrapper_stage = Usd.Stage.CreateNew(dest_path)
            rel_ref = os.path.relpath(produced_usda, start=out_dir)
            root_prim_path = Sdf.Path(f"/{robot_name}")
            prim = wrapper_stage.OverridePrim(root_prim_path)
            prim.GetReferences().ClearReferences()
            prim.GetReferences().AddReference(rel_ref)
            wrapper_stage.SetDefaultPrim(prim)
            wrapper_stage.GetRootLayer().Save()
            print(f"Wrapper USD at requested path: {dest_path} -> references {rel_ref}")
        else:
            pkg_stage = Usd.Stage.Open(produced_usda, Usd.Stage.LoadAll)
            root_prim = pkg_stage.GetDefaultPrim()
            if root_prim:
                vs = root_prim.GetVariantSets().GetVariantSet("Physics")
                if vs and "physx" in vs.GetVariantNames() and vs.GetVariantSelection() != "physx":
                    vs.SetVariantSelection("physx")

            flattened = pkg_stage.Flatten()
            if os.path.exists(dest_path):
                os.remove(dest_path)
            flattened.Export(dest_path)
            print(f"Flattened self-contained USD: {dest_path}")

            if not args_cli.no_flat_hierarchy:
                summary = _flatten_hierarchy(
                    dest_path,
                    drop_empty_scopes=not args_cli.keep_empty_scopes,
                    fix_base=args_cli.fix_base,
                )
                print(
                    f"Flat-hierarchy pass: links={summary['links']} joints={summary['joints']}"
                    f" scopes_deleted={summary['deleted_scopes']}"
                )

            # Sanity-check counts so users can confirm collisions/articulation made it in.
            try:
                from pxr import UsdGeom, UsdPhysics

                verify = Usd.Stage.Open(dest_path)
                preds = Usd.PrimAllPrimsPredicate
                n_mesh = sum(1 for p in verify.Traverse(preds) if p.IsA(UsdGeom.Mesh))
                n_coll = sum(1 for p in verify.Traverse(preds) if p.HasAPI(UsdPhysics.CollisionAPI))
                n_rb = sum(1 for p in verify.Traverse(preds) if p.HasAPI(UsdPhysics.RigidBodyAPI))
                n_art = sum(1 for p in verify.Traverse(preds) if p.HasAPI(UsdPhysics.ArticulationRootAPI))
                print(
                    f"  Mesh prims: {n_mesh} | CollisionAPI: {n_coll} | RigidBodyAPI: {n_rb} |"
                    f" ArticulationRootAPI: {n_art}"
                )
            except Exception as verify_exc:
                print(f"[WARN] post-flatten verification skipped: {verify_exc}")

            if args_cli.remove_package:
                pkg_dir = os.path.dirname(produced_usda)
                if os.path.isdir(pkg_dir) and os.path.dirname(pkg_dir) == out_dir and os.path.basename(
                    pkg_dir
                ) == robot_name:
                    shutil.rmtree(pkg_dir, ignore_errors=True)
                    print(f"Removed intermediate package directory: {pkg_dir}")
                else:
                    print(f"[WARN] refusing to remove unexpected package path: {pkg_dir}")
    except Exception as exc:
        print(f"[WARN] could not write USD at {dest_path}: {exc}")
        print(f"       use the package root directly: {produced_usda}")

    print("-" * 80)

    carb_settings_iface = carb.settings.get_settings()
    local_gui = carb_settings_iface.get("/app/window/enabled")
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    if local_gui or livestream_gui:
        try:
            sim_utils.open_stage(produced_usda)
        except AttributeError:
            omni.usd.get_context().open_stage(produced_usda)
        app = omni.kit.app.get_app_interface()
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()
