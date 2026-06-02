# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import re

import omni.kit.app

from .asset_converter_base import AssetConverterBase
from .urdf_converter_cfg import UrdfConverterCfg

_URDF_EXT = "isaacsim.asset.importer.urdf"
_JOINT_TYPES = {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint", "PhysicsFixedJoint", "PhysicsJoint"}
_COLLIDER_TYPE_MAP = {"convex_hull": "Convex Hull", "convex_decomposition": "Convex Decomposition"}


class UrdfConverter(AssetConverterBase):
    """Converter for a URDF description file to a USD file.

    This class wraps around the `isaacsim.asset.importer.urdf`_ extension to provide a lazy
    implementation for URDF to USD conversion.

    .. note::
        Isaac Sim 5.1 ships two URDF importers: the legacy ``2.4.31`` C++ extension and the
        ``3.2.1`` pure-Python rewrite. On installs where ``2.4.31`` is broken/absent (no
        ``_urdf`` binding) the kit experience pins ``3.2.1``. Stock Isaac Lab targets the
        ``2.4.31`` C++ API (``acquire_urdf_interface`` / ``URDFParseFile`` commands), which does
        not exist in ``3.2.1`` -- hence this converter is rewritten to drive ``3.2.1`` directly.

    .. note::
        The ``3.2.1`` importer emits a *nested* kinematic-tree layout (links nested under a
        ``Geometry`` scope, meshes referenced via prototypes). Isaac Lab assets are expected in a
        *flat* layout (every link a direct child of the root with a ``<link>/visuals`` mesh and a
        single articulation root on the top prim). This converter therefore flattens the importer
        output: each rigid body is reparented to the root with its **world** transform baked into
        local (so joint frames, inertias and meshes stay valid), visuals are re-referenced from the
        importer's ``instances.usda`` and the importer's ``root_joint`` fixed base is dropped unless
        :attr:`UrdfConverterCfg.fix_base` is set.

    .. caution::
        The ``3.2.1`` importer does not expose ``merge_fixed_joints`` (links joined by fixed joints
        are not consolidated). ``replace_cylinders_with_capsules`` is honored during flattening by
        retyping the importer's ``Cylinder`` colliders to ``Capsule``.

    .. _isaacsim.asset.importer.urdf: https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_urdf.html
    """

    cfg: UrdfConverterCfg
    """The configuration instance for URDF to USD conversion."""

    def __init__(self, cfg: UrdfConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for URDF to USD conversion.
        """
        # make sure the (kit-pinned) URDF importer extension is enabled
        manager = omni.kit.app.get_app().get_extension_manager()
        if not manager.is_extension_enabled(_URDF_EXT):
            manager.set_extension_enabled_immediate(_URDF_EXT, True)
        super().__init__(cfg=cfg)

    """
    Implementation specific methods.
    """

    def _convert_asset(self, cfg: UrdfConverterCfg):
        """Convert the URDF to a flat-layout USD at :attr:`usd_path`.

        Args:
            cfg: The URDF conversion configuration.
        """
        # 1) import URDF -> nested USD package using the working 3.2.1 importer
        from isaacsim.asset.importer.urdf import URDFImporterConfig

        src_dir = os.path.join(self.usd_dir, "_urdf_src")
        os.makedirs(src_dir, exist_ok=True)
        importer_cfg = URDFImporterConfig(
            urdf_path=cfg.asset_path,
            usd_path=src_dir,
            merge_mesh=False,
            collision_from_visuals=cfg.collision_from_visuals,
            collision_type=_COLLIDER_TYPE_MAP.get(cfg.collider_type, "Convex Hull"),
            allow_self_collision=cfg.self_collision,
        )
        nested_usd = self._run_importer(importer_cfg)

        # 2) flatten into the classic Isaac-Lab layout at self.usd_path. visuals are referenced
        #    from the importer's instances.usda (kept under src_dir) via a relative path.
        instances_path = os.path.join(os.path.dirname(nested_usd), "payloads", "instances.usda")
        instances_ref = os.path.relpath(instances_path, self.usd_dir)
        if not instances_ref.startswith((os.curdir + os.sep, os.pardir + os.sep, "./", "../")):
            instances_ref = "./" + instances_ref
        _flatten_to_isaaclab_layout(
            nested_usd,
            self.usd_path,
            fix_base=cfg.fix_base,
            replace_cylinders=cfg.replace_cylinders_with_capsules,
            instances_ref=instances_ref.replace(os.sep, "/"),
        )

    """
    Helper methods.
    """

    def _run_importer(self, importer_cfg) -> str:
        """Run ``URDFImporter.import_urdf``, silencing its benign per-joint warnings.

        ``import_urdf`` uses pure ``Usd.Stage.Open`` / ``Stage.Export`` (it does not touch the
        omni.usd context), so it is safe to call inline during scene creation. It does, however,
        emit a Python-logging "stiffness and damping not available" warning for every drive-less
        joint (harmless -- Isaac Lab's actuators set the gains at runtime); raise those loggers to
        ERROR for the duration of the import.
        """
        import logging

        from isaacsim.asset.importer.urdf import URDFImporter

        noisy = (
            "isaacsim.asset.importer.utils.impl.urdf_to_mjc_physx_conversion_utils",
            "isaacsim.asset.importer.utils.impl.mjc_to_physx_conversion_utils",
        )
        saved = {n: logging.getLogger(n).level for n in noisy}
        for n in noisy:
            logging.getLogger(n).setLevel(logging.ERROR)
        try:
            return URDFImporter(importer_cfg).import_urdf(importer_cfg)
        finally:
            for n, lvl in saved.items():
                logging.getLogger(n).setLevel(lvl)


"""
Flattening: nested 3.2.1 importer output -> classic flat Isaac-Lab layout.
"""


def _strip_articulation_root(layer, path) -> None:
    """Drop any *ArticulationRoot* applied API from a copied body prim spec."""
    from pxr import Sdf

    spec = layer.GetPrimAtPath(path)
    if spec is None or not spec.HasInfo("apiSchemas"):
        return
    listop = spec.GetInfo("apiSchemas")
    items = [s for s in listop.GetAddedOrExplicitItems() if "ArticulationRoot" not in s]
    new = Sdf.TokenListOp()
    new.prependedItems = items
    spec.SetInfo("apiSchemas", new)


def _instance_names_for(link: str, available: set) -> list:
    """Visual instance prim names in instances.usda for ``link`` (``link`` and ``link_<n>``)."""
    pat = re.compile(rf"^{re.escape(link)}(_\d+)?$")
    return sorted(n for n in available if pat.match(n))


def _flatten_to_isaaclab_layout(
    nested_usd: str, out_usd: str, *, fix_base: bool, replace_cylinders: bool, instances_ref: str
) -> None:
    """Flatten the nested importer output into the classic flat Isaac-Lab layout at ``out_usd``."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    src = Usd.Stage.Open(nested_usd)
    flat_layer = src.Flatten()  # merge variants/payloads/overs so bodies carry their physics specs
    src = Usd.Stage.Open(flat_layer)

    root_name = src.GetDefaultPrim().GetName()
    root_path = f"/{root_name}"

    # geometry instances available in instances.usda (re-referenced for each link's visuals)
    inst_stage = Usd.Stage.Open(os.path.join(os.path.dirname(nested_usd), "payloads", "instances.usda"))
    inst_root = inst_stage.GetPrimAtPath("/Instances")
    available = {c.GetName() for c in inst_root.GetChildren()} if inst_root else set()

    # rigid bodies (the link Xforms) and their world transforms
    bodies = {}
    for prim in src.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies[prim.GetName()] = prim.GetPath()
    world_xf = {
        name: UsdGeom.Xformable(src.GetPrimAtPath(p)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for name, p in bodies.items()
    }
    # joints (drop the importer's fixed base 'root_joint' unless a fixed base is requested)
    drop = set() if fix_base else {"root_joint"}
    joints = [p for p in src.Traverse() if p.GetTypeName() in _JOINT_TYPES and p.GetName() not in drop]

    if os.path.exists(out_usd):
        os.remove(out_usd)
    out_layer = Sdf.Layer.CreateNew(out_usd)
    out = Usd.Stage.Open(out_layer)
    UsdGeom.SetStageUpAxis(out, UsdGeom.GetStageUpAxis(src))
    UsdGeom.SetStageMetersPerUnit(out, UsdGeom.GetStageMetersPerUnit(src) or 1.0)

    root = UsdGeom.Xform.Define(out, root_path)
    out.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    for name, sp in bodies.items():
        tp = Sdf.Path(f"{root_path}/{name}")
        Sdf.CopySpec(flat_layer, sp, out_layer, tp)  # brings the body's physics schemas + attrs
        tprim = out.GetPrimAtPath(tp)

        # make the body a plain physics Xform: drop its own geometry reference/instancing
        spec = out_layer.GetPrimAtPath(tp)
        spec.referenceList.ClearEdits()
        spec.ClearInstanceable()
        _strip_articulation_root(out_layer, tp)

        # bake world transform into local (parent is the identity root)
        xf = UsdGeom.Xformable(tprim)
        xf.ClearXformOpOrder()
        for attr in list(tprim.GetAttributes()):
            if attr.GetName().startswith("xformOp:"):
                tprim.RemoveProperty(attr.GetName())
        xf.AddTransformOp().Set(Gf.Matrix4d(world_xf[name]))

        # keep only collision children; drop child-link subtrees and stale visual instances.
        # honor replace_cylinders_with_capsules by retyping Cylinder colliders to Capsule (both
        # share the radius/height/axis schema, so the geometry carries over unchanged).
        for child in list(tprim.GetChildren()):
            if not child.HasAPI(UsdPhysics.CollisionAPI):
                out.RemovePrim(child.GetPath())
            elif replace_cylinders and child.GetTypeName() == "Cylinder":
                child.SetTypeName("Capsule")

        # re-add visuals as references to the intact instances.usda (preserves mesh + materials)
        inst_names = _instance_names_for(name, available)
        if inst_names:
            UsdGeom.Xform.Define(out, tp.AppendChild("visuals"))
            for inst in inst_names:
                vprim = out.DefinePrim(tp.AppendPath(Sdf.Path(f"visuals/{inst}")), "Xform")
                vprim.GetReferences().AddReference(instances_ref, f"/Instances/{inst}")
                vprim.SetInstanceable(True)

    # joints under /<root>/joints with body0/body1 retargeted to the flat paths
    UsdGeom.Scope.Define(out, f"{root_path}/joints")
    for j in joints:
        jp = Sdf.Path(f"{root_path}/joints/{j.GetName()}")
        Sdf.CopySpec(flat_layer, j.GetPath(), out_layer, jp)
        jprim = out.GetPrimAtPath(jp)
        for rel_name in ("physics:body0", "physics:body1"):
            rel = jprim.GetRelationship(rel_name)
            if rel and rel.GetTargets():
                rel.SetTargets(
                    [
                        Sdf.Path(root_path) if t.name == root_name else Sdf.Path(f"{root_path}/{t.name}")
                        for t in rel.GetTargets()
                    ]
                )

    out_layer.Save()
