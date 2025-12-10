from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
):
    """刚体缩放随机化事件函数（在 USD 级别修改 xformOp:scale）。"""
    """Randomize the scale of a rigid body asset in the USD stage.

    This function modifies the "xformOp:scale" property of all the prims corresponding to the asset.

    It takes a tuple or dictionary for the scale ranges. If it is a tuple, then the scaling along
    individual axis is performed equally. If it is a dictionary, the scaling is independent across each dimension.
    The keys of the dictionary are ``x``, ``y``, and ``z``. The values are tuples of the form ``(min, max)``.

    If the dictionary does not contain a key, the range is set to one for that axis.

    Relative child path can be used to randomize the scale of a specific child prim of the asset.
    For example, if the asset at prim path expression ``/World/envs/env_.*/Object`` has a child
    with the path ``/World/envs/env_.*/Object/mesh``, then the relative child path should be ``mesh`` or
    ``/mesh``.

    .. attention::
        Since this function modifies USD properties that are parsed by the physics engine once the simulation
        starts, the term should only be used before the simulation starts playing. This corresponds to the
        event mode named "usd". Using it at simulation time, may lead to unpredictable behaviors.

    .. note::
        When randomizing the scale of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """
    # 检查仿真是否已经在运行（只能在播放前修改 USD 属性）
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'usd' mode."
        )

    # 提取场景中的刚体对象，方便类型提示
    asset: RigidObject = env.scene[asset_cfg.name]

    if isinstance(asset, Articulation):
        raise ValueError(
            "Scaling an articulation randomly is not supported, as it affects joint attributes and can cause"
            " unexpected behavior. To achieve different scales, we recommend generating separate USD files for"
            " each version of the articulation and using multi-asset spawning. For more details, refer to:"
            " https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html"
        )

    # 解析需要随机化的环境 ID；为空则对全部环境生效
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # 获取当前 USD Stage
    stage = get_current_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    # 采样缩放系数（支持统一缩放或 xyz 分量独立缩放）
    if isinstance(scale_range, dict):
        range_list = [scale_range.get(key, (1.0, 1.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device="cpu")
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")
    else:
        rand_samples = math_utils.sample_uniform(*scale_range, (len(env_ids), 1), device="cpu")
        rand_samples = rand_samples.repeat(1, 3)
    # 转成 Python list，便于在 for 循环中按环境索引
    rand_samples = rand_samples.tolist()

    # 若未提供子路径，则默认对 asset 根 prim 生效
    # 若提供相对子路径，可只随机化层级中的某个子 mesh
    if relative_child_path is None:
        relative_child_path = ""
    elif not relative_child_path.startswith("/"):
        relative_child_path = "/" + relative_child_path

    # 使用 Sdf.ChangeBlock() 批量修改 USD，提升写入效率
    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            # 当前环境下需要随机化的 prim 路径
            prim_path = prim_paths[env_id] + relative_child_path
            # 在 root layer 中确保 prim spec 存在
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # 获取/创建缩放属性 xformOp:scale
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            # 若不存在缩放属性，则新建一个 Double3 类型的 AttributeSpec
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

            # 写入随机缩放值
            scale_spec.default = Gf.Vec3f(*rand_samples[i])

            # 如果新建了 scale 属性，则需要保证变换栈顺序正确
            # 默认顺序为 translate -> orient -> scale，与 Isaac Sim 约定保持一致
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])


class randomize_rigid_body_material(ManagerTermBase):
    """Randomize the physics materials on all geometries of the asset.

    This function creates a set of physics materials with random static friction, dynamic friction, and restitution
    values. The number of materials is specified by ``num_buckets``. The materials are generated by sampling
    uniform random values from the given ranges.

    The material properties are then assigned to the geometries of the asset. The assignment is done by
    creating a random integer tensor of shape  (num_instances, max_num_shapes) where ``num_instances``
    is the number of assets spawned and ``max_num_shapes`` is the maximum number of shapes in the asset (over
    all bodies). The integer values are used as indices to select the material properties from the
    material buckets.

    If the flag ``make_consistent`` is set to ``True``, the dynamic friction is set to be less than or equal to
    the static friction. This obeys the physics constraint on friction values. However, it may not always be
    essential for the application. Thus, the flag is set to ``False`` by default.

    .. attention::
        This function uses CPU tensors to assign the material properties. It is recommended to use this function
        only during the initialization of the environment. Otherwise, it may lead to a significant performance
        overhead.

    .. note::
        PhysX only allows 64000 unique physics materials in the scene. If the number of materials exceeds this
        limit, the simulation will crash. Due to this reason, we sample the materials only once during initialization.
        Afterwards, these materials are randomly assigned to the geometries of the asset.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)

        # 从配置中提取资产配置与实例，便于类型提示与后续访问
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        # 仅支持刚体对象或关节结构，其他类型直接报错
        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        # 计算每个刚体上的 shape 数量，用于正确索引材质（Articulation 无直接接口，只能通过 PhysX 视图推断）
        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
            # 校验解析到的 shape 总数与预期是否一致
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = self.asset.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            # in this case, we don't need to do special indexing
            self.num_shapes_per_body = None

        # 从 cfg 中取得摩擦系数与恢复系数采样范围与桶数
        static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        dynamic_friction_range = cfg.params.get("dynamic_friction_range", (1.0, 1.0))
        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
        num_buckets = int(cfg.params.get("num_buckets", 1))

        # 在初始化阶段一次性采样 num_buckets 组材质参数，后续只做索引分配
        range_list = [static_friction_range, dynamic_friction_range, restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        self.material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")

        # 若需要，强制动态摩擦不大于静摩擦，保证物理合理性
        make_consistent = cfg.params.get("make_consistent", False)
        if make_consistent:
            self.material_buckets[:, 1] = torch.min(self.material_buckets[:, 0], self.material_buckets[:, 1])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        # 解析环境 ID；为空则对所有环境随机化
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # 为每个环境、每个 shape 随机分配一个材质桶索引
        total_num_shapes = self.asset.root_physx_view.max_shapes
        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), total_num_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids]

        # 从物理仿真中取出现有材质缓冲区
        materials = self.asset.root_physx_view.get_material_properties()

        # 用采样到的新材质覆盖缓冲区
        if self.num_shapes_per_body is not None:
            # 对 Articulation，按 body 逐段写入对应的 shape 区间
            for body_id in self.asset_cfg.body_ids:
                # 计算该 body 对应的 shape 索引范围
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                # assign the new materials
                # material samples are of shape: num_env_ids x total_num_shapes x 3
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            # 否则直接整体覆盖所有 shape 的材质
            materials[env_ids] = material_samples[:]

        # 将更新后的材质缓冲区写回物理仿真
        self.asset.root_physx_view.set_material_properties(materials, env_ids)


class randomize_rigid_body_mass(ManagerTermBase):
    """Randomize the mass of the bodies by adding, scaling, or setting random values.

    This function allows randomizing the mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the operation.

    If the ``recompute_inertia`` flag is set to ``True``, the function recomputes the inertia tensor of the bodies
    after setting the mass. This is useful when the mass is changed significantly, as the inertia tensor depends
    on the mass. It assumes the body is a uniform density object. If the body is not a uniform density object,
    the inertia tensor may not be accurate.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        super().__init__(cfg, env)

        # 从配置中提取资产和随机化参数
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # 校验 operation 是否有效
        if cfg.params["operation"] == "scale":
            if "mass_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["mass_distribution_params"], "mass_distribution_params", allow_zero=False
                )
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_rigid_body_mass' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        mass_distribution_params: tuple[float, float],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        recompute_inertia: bool = True,
    ):
        # 解析环境 ID；为空则对所有环境随机化
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # 解析需要随机化的 body 索引；slice(None) 表示所有刚体
        if self.asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

        # 从 PhysX 视图获取当前质量矩阵 (num_envs, num_bodies)
        masses = self.asset.root_physx_view.get_masses()

        # 始终在默认质量的基础上重新随机，避免多次调用时叠加误差
        masses[env_ids[:, None], body_ids] = self.asset.data.default_mass[env_ids[:, None], body_ids].clone()

        # 按指定分布与操作对质量进行随机化（加法 / 缩放 / 绝对赋值）
        masses = _randomize_prop_by_op(
            masses, mass_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
        )

        # 将随机后的质量写回物理仿真
        self.asset.root_physx_view.set_masses(masses, env_ids)

        # 如有需要，按质量变化比例缩放惯性张量
        if recompute_inertia:
            # compute the ratios of the new masses to the initial masses
            ratios = masses[env_ids[:, None], body_ids] / self.asset.data.default_mass[env_ids[:, None], body_ids]
            # 按比例缩放默认惯性张量（质量随机化始终基于默认值）
            inertias = self.asset.root_physx_view.get_inertias()
            if isinstance(self.asset, Articulation):
                # inertia has shape: (num_envs, num_bodies, 9) for articulation
                inertias[env_ids[:, None], body_ids] = (
                    self.asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
                )
            else:
                # inertia has shape: (num_envs, 9) for rigid object
                inertias[env_ids] = self.asset.data.default_inertia[env_ids] * ratios
            # 将新的惯性张量写回物理仿真
            self.asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # 从场景中获取对应的关节结构，用于修改质心
    asset: Articulation = env.scene[asset_cfg.name]
    # 解析环境 ID；为空则对所有环境生效
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # 按 xyz 范围采样质心偏移量
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # 从 PhysX 视图获取当前质心
    coms = asset.root_physx_view.get_coms().clone()

    # 在采样范围内平移质心
    coms[env_ids[:, None], body_ids, :3] += rand_samples

    # 将新的质心写回物理仿真
    asset.root_physx_view.set_coms(coms, env_ids)


class randomize_rigid_body_inertia(ManagerTermBase):
    """Randomize the inertia tensor of rigid bodies by adding, scaling, or setting random values.

    This function allows randomizing the inertia tensor of the bodies of the asset independently
    from mass randomization. The function samples random values from the given distribution
    parameters and adds, scales, or sets the values into the physics simulation based on the operation.

    The same scale factor is applied to all 9 components of the inertia tensor for each body,
    preserving the relative shape of the inertia tensor while changing its magnitude.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        super().__init__(cfg, env)

        # 从配置中提取资产和随机化参数
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # 校验 operation 是否有效
        if cfg.params["operation"] == "scale":
            if "inertia_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["inertia_distribution_params"], "inertia_distribution_params", allow_zero=False
                )
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_rigid_body_inertia' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg = None,
        inertia_distribution_params: tuple[float, float] = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # 解析环境 ID；为空则对所有环境随机化
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # 解析需要随机化的 body 索引；slice(None) 表示所有刚体
        if self.asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

        # 从 PhysX 视图获取当前惯性张量
        inertias = self.asset.root_physx_view.get_inertias()

        # 选择采样分布函数
        if distribution == "uniform":
            dist_fn = math_utils.sample_uniform
        elif distribution == "log_uniform":
            dist_fn = math_utils.sample_log_uniform
        elif distribution == "gaussian":
            dist_fn = math_utils.sample_gaussian
        else:
            raise NotImplementedError(
                f"Unknown distribution: '{distribution}' for inertia randomization."
                " Please use 'uniform', 'log_uniform', 'gaussian'."
            )

        # 始终在默认惯性的基础上重新随机，避免多次调用时叠加误差
        if isinstance(self.asset, Articulation):
            # inertia has shape: (num_envs, num_bodies, 9) for articulation
            inertias[env_ids[:, None], body_ids] = self.asset.data.default_inertia[env_ids[:, None], body_ids].clone()
            # 为每个 (env, body) 采样一个缩放因子，然后广播到 9 个惯性分量
            n_envs = len(env_ids)
            n_bodies = len(body_ids)
            scale_factors = dist_fn(*inertia_distribution_params, (n_envs, n_bodies), device="cpu")
            # 将缩放因子扩展到 (n_envs, n_bodies, 9)
            scale_factors = scale_factors.unsqueeze(-1).expand(-1, -1, 9)

            if operation == "scale":
                inertias[env_ids[:, None], body_ids] *= scale_factors
            elif operation == "add":
                inertias[env_ids[:, None], body_ids] += scale_factors
            else:  # abs
                inertias[env_ids[:, None], body_ids] = scale_factors
        else:
            # inertia has shape: (num_envs, 9) for rigid object
            inertias[env_ids] = self.asset.data.default_inertia[env_ids].clone()
            n_envs = len(env_ids)
            scale_factors = dist_fn(*inertia_distribution_params, (n_envs, 1), device="cpu")
            # 将缩放因子扩展到 (n_envs, 9)
            scale_factors = scale_factors.expand(-1, 9)

            if operation == "scale":
                inertias[env_ids] *= scale_factors
            elif operation == "add":
                inertias[env_ids] += scale_factors
            else:  # abs
                inertias[env_ids] = scale_factors

        # 确保惯性张量非负
        inertias = torch.clamp(inertias, min=1e-9)

        # 将新的惯性张量写回物理仿真
        self.asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_rigid_body_collider_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    rest_offset_distribution_params: tuple[float, float] | None = None,
    contact_offset_distribution_params: tuple[float, float] | None = None,
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the collider parameters of rigid bodies in an asset by adding, scaling, or setting random values.

    This function allows randomizing the collider parameters of the asset, such as rest and contact offsets.
    These correspond to the physics engine collider properties that affect the collision checking.

    The function samples random values from the given distribution parameters and applies the operation to
    the collider properties. It then sets the values into the physics simulation. If the distribution parameters
    are not provided for a particular property, the function does not modify the property.

    Currently, the distribution parameters are applied as absolute values.

    .. tip::
        This function uses CPU tensors to assign the collision properties. It is recommended to use this function
        only during the initialization of the environment.
    """
    # 从场景中获取刚体 / 关节结构
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")

    # 按给定范围采样并写入碰撞体属性
    # -- rest offsets（休止偏移）
    if rest_offset_distribution_params is not None:
        rest_offset = asset.root_physx_view.get_rest_offsets().clone()
        rest_offset = _randomize_prop_by_op(
            rest_offset,
            rest_offset_distribution_params,
            None,
            slice(None),
            operation="abs",
            distribution=distribution,
        )
        asset.root_physx_view.set_rest_offsets(rest_offset, env_ids.cpu())
    # -- contact offsets（接触偏移）
    if contact_offset_distribution_params is not None:
        contact_offset = asset.root_physx_view.get_contact_offsets().clone()
        contact_offset = _randomize_prop_by_op(
            contact_offset,
            contact_offset_distribution_params,
            None,
            slice(None),
            operation="abs",
            distribution=distribution,
        )
        asset.root_physx_view.set_contact_offsets(contact_offset, env_ids.cpu())


def randomize_physics_scene_gravity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    gravity_distribution_params: tuple[list[float], list[float]],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize gravity by adding, scaling, or setting random values.

    This function allows randomizing gravity of the physics scene. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the
    operation.

    The distribution parameters are lists of two elements each, representing the lower and upper bounds of the
    distribution for the x, y, and z components of the gravity vector. The function samples random values for each
    component independently.

    .. attention::
        This function applied the same gravity for all the environments.

    .. tip::
        This function uses CPU tensors to assign gravity.
    """
    # 从环境配置中读取当前重力向量
    gravity = torch.tensor(env.sim.cfg.gravity, device="cpu").unsqueeze(0)
    dist_param_0 = torch.tensor(gravity_distribution_params[0], device="cpu")
    dist_param_1 = torch.tensor(gravity_distribution_params[1], device="cpu")
    # 对重力向量按指定分布与操作进行随机化
    gravity = _randomize_prop_by_op(
        gravity,
        (dist_param_0, dist_param_1),
        None,
        slice(None),
        operation=operation,
        distribution=distribution,
    )
    # 去掉 batch 维，转成 Python list 以便写回引擎
    gravity = gravity[0].tolist()

    # 将新的重力写入 PhysX 场景
    physics_sim_view: physx.SimulationView = sim_utils.SimulationContext.instance().physics_sim_view
    physics_sim_view.set_gravity(carb.Float3(*gravity))


class randomize_actuator_gains(ManagerTermBase):
    """Randomize the actuator gains in an articulation by adding, scaling, or setting random values.

    This function allows randomizing the actuator stiffness and damping gains.

    The function samples random values from the given distribution parameters and applies the operation to the joint properties.
    It then sets the values into the actuator models. If the distribution parameters are not provided for a particular property,
    the function does not modify the property.

    .. tip::
        For implicit actuators, this function uses CPU tensors to assign the actuator gains into the simulation.
        In such cases, it is recommended to use this function only during the initialization of the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        super().__init__(cfg, env)

        # 从配置中提取目标资产（包含一组执行器）
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # 校验 operation 是否有效（主要影响缩放范围是否合法）
        if cfg.params["operation"] == "scale":
            if "stiffness_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["stiffness_distribution_params"], "stiffness_distribution_params", allow_zero=False
                )
            if "damping_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["damping_distribution_params"], "damping_distribution_params")
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_actuator_gains' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # 解析环境 ID；为空则对所有环境生效
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        def randomize(data: torch.Tensor, params: tuple[float, float]) -> torch.Tensor:
            # 对给定张量的选择维度进行随机化（加法 / 缩放 / 绝对值）
            return _randomize_prop_by_op(
                data, params, dim_0_ids=None, dim_1_ids=actuator_indices, operation=operation, distribution=distribution
            )

        # 遍历该资产下的所有执行器，按配置随机化刚度/阻尼
        for actuator in self.asset.actuators.values():
            if isinstance(self.asset_cfg.joint_ids, slice):
                # 配置为 slice(None) 时，执行器下的所有关节都参与随机化
                actuator_indices = slice(None)
                if isinstance(actuator.joint_indices, slice):
                    global_indices = slice(None)
                elif isinstance(actuator.joint_indices, torch.Tensor):
                    global_indices = actuator.joint_indices.to(self.asset.device)
                else:
                    raise TypeError("Actuator joint indices must be a slice or a torch.Tensor.")
            elif isinstance(actuator.joint_indices, slice):
                # 执行器 joint_indices 是 slice，按场景配置中的 joint_ids 取子集
                global_indices = actuator_indices = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
            else:
                # 其他情况：取执行器关节与配置关节的交集
                actuator_joint_indices = actuator.joint_indices
                asset_joint_ids = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
                # actuator 内部需要随机化的局部索引
                actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
                if len(actuator_indices) == 0:
                    continue
                # 将局部索引映射到全局关节索引
                global_indices = actuator_joint_indices[actuator_indices]
            # Randomize stiffness
            if stiffness_distribution_params is not None:
                # 先用默认关节刚度覆盖，再在其上做随机化
                stiffness = actuator.stiffness[env_ids].clone()
                stiffness[:, actuator_indices] = self.asset.data.default_joint_stiffness[env_ids][
                    :, global_indices
                ].clone()
                randomize(stiffness, stiffness_distribution_params)
                actuator.stiffness[env_ids] = stiffness
                if isinstance(actuator, ImplicitActuator):
                    # 隐式执行器需要将新的刚度写回到底层模拟
                    self.asset.write_joint_stiffness_to_sim(
                        stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids
                    )
            # Randomize damping
            if damping_distribution_params is not None:
                # 同理，对关节阻尼进行随机化
                damping = actuator.damping[env_ids].clone()
                damping[:, actuator_indices] = self.asset.data.default_joint_damping[env_ids][:, global_indices].clone()
                randomize(damping, damping_distribution_params)
                actuator.damping[env_ids] = damping
                if isinstance(actuator, ImplicitActuator):
                    # 将随机后的阻尼写入模拟
                    self.asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)


class randomize_joint_parameters(ManagerTermBase):
    """Randomize the simulated joint parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the joint parameters of the asset. These correspond to the physics engine
    joint properties that affect the joint behavior. The properties include the joint friction coefficient, armature,
    and joint position limits.

    The function samples random values from the given distribution parameters and applies the operation to the
    joint properties. It then sets the values into the physics simulation. If the distribution parameters are
    not provided for a particular property, the function does not modify the property.

    .. tip::
        This function uses CPU tensors to assign the joint properties. It is recommended to use this function
        only during the initialization of the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        super().__init__(cfg, env)

        # 记录目标资产及配置，用于后续关节属性随机化
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # 校验 operation 与缩放范围是否合法
        if cfg.params["operation"] == "scale":
            if "friction_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["friction_distribution_params"], "friction_distribution_params")
            if "armature_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["armature_distribution_params"], "armature_distribution_params")
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_fixed_tendon_parameters' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg = None,
        friction_distribution_params: tuple[float, float] | None = None,
        armature_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # 解析环境 ID
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        # 根据配置解析参与随机化的关节索引
        if self.asset_cfg.joint_ids == slice(None):
            joint_ids = slice(None)  # for optimization purposes
        else:
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, dtype=torch.int, device=self.asset.device)

        # 根据配置对各关节属性采样并写入物理仿真
        # 关节摩擦系数
        if friction_distribution_params is not None:
            friction_coeff = _randomize_prop_by_op(
                self.asset.data.default_joint_friction_coeff.clone(),
                friction_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )

            # ensure the friction coefficient is non-negative
            friction_coeff = torch.clamp(friction_coeff, min=0.0)

            # Always set static friction (indexed once)
            static_friction_coeff = friction_coeff[env_ids[:, None], joint_ids]

            # if isaacsim version is lower than 5.0.0 we can set only the static friction coefficient
            major_version = int(env.sim.get_version()[0])
            if major_version >= 5:
                # Randomize raw tensors
                dynamic_friction_coeff = _randomize_prop_by_op(
                    self.asset.data.default_joint_dynamic_friction_coeff.clone(),
                    friction_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
                viscous_friction_coeff = _randomize_prop_by_op(
                    self.asset.data.default_joint_viscous_friction_coeff.clone(),
                    friction_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )

                # Clamp to non-negative
                dynamic_friction_coeff = torch.clamp(dynamic_friction_coeff, min=0.0)
                viscous_friction_coeff = torch.clamp(viscous_friction_coeff, min=0.0)

                # Ensure dynamic ≤ static (same shape before indexing)
                dynamic_friction_coeff = torch.minimum(dynamic_friction_coeff, friction_coeff)

                # Index once at the end
                dynamic_friction_coeff = dynamic_friction_coeff[env_ids[:, None], joint_ids]
                viscous_friction_coeff = viscous_friction_coeff[env_ids[:, None], joint_ids]
            else:
                # For versions < 5.0.0, we do not set these values
                dynamic_friction_coeff = None
                viscous_friction_coeff = None

            # Single write call for all versions
            self.asset.write_joint_friction_coefficient_to_sim(
                joint_friction_coeff=static_friction_coeff,
                joint_dynamic_friction_coeff=dynamic_friction_coeff,
                joint_viscous_friction_coeff=viscous_friction_coeff,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )

        # joint armature
        if armature_distribution_params is not None:
            armature = _randomize_prop_by_op(
                self.asset.data.default_joint_armature.clone(),
                armature_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            # extract the armature for the concerned joints
            if isinstance(joint_ids, slice):
                armature_to_write = armature[env_ids]
            else:
                armature_to_write = armature[env_ids[:, None], joint_ids]
            self.asset.write_joint_armature_to_sim(
                armature_to_write, joint_ids=joint_ids, env_ids=env_ids
            )

        # joint position limits
        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            joint_pos_limits = self.asset.data.default_joint_pos_limits.clone()
            # -- randomize the lower limits
            if lower_limit_distribution_params is not None:
                joint_pos_limits[..., 0] = _randomize_prop_by_op(
                    joint_pos_limits[..., 0],
                    lower_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
            # -- randomize the upper limits
            if upper_limit_distribution_params is not None:
                joint_pos_limits[..., 1] = _randomize_prop_by_op(
                    joint_pos_limits[..., 1],
                    upper_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )

            # extract the position limits for the concerned joints
            joint_pos_limits = joint_pos_limits[env_ids[:, None], joint_ids]
            if (joint_pos_limits[..., 0] > joint_pos_limits[..., 1]).any():
                raise ValueError(
                    "Randomization term 'randomize_joint_parameters' is setting lower joint limits that are greater"
                    " than upper joint limits. Please check the distribution parameters for the joint position limits."
                )
            # set the position limits into the physics simulation
            self.asset.write_joint_position_limit_to_sim(
                joint_pos_limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
            )


class randomize_fixed_tendon_parameters(ManagerTermBase):
    """Randomize the simulated fixed tendon parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the fixed tendon parameters of the asset.
    These correspond to the physics engine tendon properties that affect the joint behavior.

    The function samples random values from the given distribution parameters and applies the operation to the tendon properties.
    It then sets the values into the physics simulation. If the distribution parameters are not provided for a
    particular property, the function does not modify the property.

    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "stiffness_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["stiffness_distribution_params"], "stiffness_distribution_params", allow_zero=False
                )
            if "damping_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["damping_distribution_params"], "damping_distribution_params")
            if "limit_stiffness_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["limit_stiffness_distribution_params"], "limit_stiffness_distribution_params"
                )
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_fixed_tendon_parameters' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        limit_stiffness_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        rest_length_distribution_params: tuple[float, float] | None = None,
        offset_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        # resolve joint indices
        if self.asset_cfg.fixed_tendon_ids == slice(None):
            tendon_ids = slice(None)  # for optimization purposes
        else:
            tendon_ids = torch.tensor(self.asset_cfg.fixed_tendon_ids, dtype=torch.int, device=self.asset.device)

        # sample tendon properties from the given ranges and set into the physics simulation
        # stiffness
        if stiffness_distribution_params is not None:
            stiffness = _randomize_prop_by_op(
                self.asset.data.default_fixed_tendon_stiffness.clone(),
                stiffness_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_stiffness(stiffness[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

        # damping
        if damping_distribution_params is not None:
            damping = _randomize_prop_by_op(
                self.asset.data.default_fixed_tendon_damping.clone(),
                damping_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_damping(damping[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

        # limit stiffness
        if limit_stiffness_distribution_params is not None:
            limit_stiffness = _randomize_prop_by_op(
                self.asset.data.default_fixed_tendon_limit_stiffness.clone(),
                limit_stiffness_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_limit_stiffness(
                limit_stiffness[env_ids[:, None], tendon_ids], tendon_ids, env_ids
            )

        # position limits
        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            limit = self.asset.data.default_fixed_tendon_pos_limits.clone()
            # -- lower limit
            if lower_limit_distribution_params is not None:
                limit[..., 0] = _randomize_prop_by_op(
                    limit[..., 0],
                    lower_limit_distribution_params,
                    env_ids,
                    tendon_ids,
                    operation=operation,
                    distribution=distribution,
                )
            # -- upper limit
            if upper_limit_distribution_params is not None:
                limit[..., 1] = _randomize_prop_by_op(
                    limit[..., 1],
                    upper_limit_distribution_params,
                    env_ids,
                    tendon_ids,
                    operation=operation,
                    distribution=distribution,
                )

            # check if the limits are valid
            tendon_limits = limit[env_ids[:, None], tendon_ids]
            if (tendon_limits[..., 0] > tendon_limits[..., 1]).any():
                raise ValueError(
                    "Randomization term 'randomize_fixed_tendon_parameters' is setting lower tendon limits that are"
                    " greater than upper tendon limits."
                )
            self.asset.set_fixed_tendon_position_limit(tendon_limits, tendon_ids, env_ids)

        # rest length
        if rest_length_distribution_params is not None:
            rest_length = _randomize_prop_by_op(
                self.asset.data.default_fixed_tendon_rest_length.clone(),
                rest_length_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_rest_length(rest_length[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

        # offset
        if offset_distribution_params is not None:
            offset = _randomize_prop_by_op(
                self.asset.data.default_fixed_tendon_offset.clone(),
                offset_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_offset(offset[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

        # write the fixed tendon properties into the simulation
        self.asset.write_fixed_tendon_properties_to_sim(tendon_ids, env_ids)


def apply_external_force_torque(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(env_ids), num_bodies, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device)
    torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)


def push_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_with_random_orientation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root position and velocities sampled randomly within the given ranges
    and the asset root orientation sampled randomly from the SO(3).

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation uniformly from the SO(3) and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of position ranges for each axis. The keys of the dictionary are ``x``,
      ``y``, and ``z``. The orientation is sampled uniformly from the SO(3).
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples
    orientations = math_utils.random_orientation(len(env_ids), device=asset.device)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_from_terrain(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state by sampling a random valid pose from the terrain.

    This function samples a random valid pose(based on flat patches) from the terrain and sets the root state
    of the asset to this position. The function also samples random velocities from the given ranges and sets them
    into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of pose ranges for each axis. The keys of the dictionary are ``roll``,
      ``pitch``, and ``yaw``. The position is sampled from the flat patches of the terrain.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.

    Note:
        The function expects the terrain to have valid flat patches under the key "init_pos". The flat patches
        are used to sample the random pose for the robot.

    Raises:
        ValueError: If the terrain does not have valid flat patches under the key "init_pos".
    """
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # obtain all flat patches corresponding to the valid poses
    valid_positions: torch.Tensor = terrain.flat_patches.get("init_pos")
    if valid_positions is None:
        raise ValueError(
            "The event term 'reset_root_state_from_terrain' requires valid flat patches under 'init_pos'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )

    # sample random valid poses
    ids = torch.randint(0, valid_positions.shape[2], size=(len(env_ids),), device=env.device)
    positions = valid_positions[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], ids]
    positions += asset.data.default_root_state[env_ids, :3]

    # sample random orientations
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # convert to quaternions
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = asset.data.default_root_state[env_ids, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # get default joint state
    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    # scale these values randomly
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def reset_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # get default joint state
    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def reset_nodal_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset nodal state to a random position and velocity uniformly within the given ranges.

    This function randomizes the nodal position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default nodal position, before setting
      them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis. The keys of the
    dictionary are ``x``, ``y``, ``z``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: DeformableObject = env.scene[asset_cfg.name]
    # get default root state
    nodal_state = asset.data.default_nodal_state_w[env_ids].clone()

    # position
    range_list = [position_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 1, 3), device=asset.device)

    nodal_state[..., :3] += rand_samples

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 1, 3), device=asset.device)

    nodal_state[..., 3:] += rand_samples

    # set into the physics simulation
    asset.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor, reset_joint_targets: bool = False):
    """Reset the scene to the default state specified in the scene configuration.

    If :attr:`reset_joint_targets` is True, the joint position and velocity targets of the articulations are
    also reset to their default values. This might be useful for some cases to clear out any previously set targets.
    However, this is not the default behavior as based on our experience, it is not always desired to reset
    targets to default values, especially when the targets should be handled by action terms and not event terms.
    """
    # rigid bodies
    for rigid_object in env.scene.rigid_objects.values():
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        rigid_object.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
    # articulations
    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        # reset joint targets if required
        if reset_joint_targets:
            articulation_asset.set_joint_position_target(default_joint_pos, env_ids=env_ids)
            articulation_asset.set_joint_velocity_target(default_joint_vel, env_ids=env_ids)
    # deformable objects
    for deformable_object in env.scene.deformable_objects.values():
        # obtain default and set into the physics simulation
        nodal_state = deformable_object.data.default_nodal_state_w[env_ids].clone()
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


class randomize_visual_texture_material(ManagerTermBase):
    """Randomize the visual texture of bodies on an asset using Replicator API.

    This function randomizes the visual texture of the bodies of the asset using the Replicator API.
    The function samples random textures from the given texture paths and applies them to the bodies
    of the asset. The textures are projected onto the bodies and rotated by the given angles.

    .. note::
        The function assumes that the asset follows the prim naming convention as:
        "{asset_prim_path}/{body_name}/visuals" where the body name is the name of the body to
        which the texture is applied. This is the default prim ordering when importing assets
        from the asset converters in Isaac Lab.

    .. note::
        When randomizing the texture of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # check to make sure replicate_physics is set to False, else raise error
        # note: We add an explicit check here since texture randomization can happen outside of 'prestartup' mode
        #   and the event manager doesn't check in that case.
        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "Unable to randomize visual texture material with scene replication enabled."
                " For stable USD-level randomization, please disable scene replication"
                " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
            )

        # enable replicator extension if not already enabled
        enable_extension("omni.replicator.core")

        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # read parameters from the configuration
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")

        # obtain the asset entity
        asset = env.scene[asset_cfg.name]

        # join all bodies in the asset
        body_names = asset_cfg.body_names
        if isinstance(body_names, str):
            body_names_regex = body_names
        elif isinstance(body_names, list):
            body_names_regex = "|".join(body_names)
        else:
            body_names_regex = ".*"

        # create the affected prim path
        # Check if the pattern with '/visuals' yields results when matching `body_names_regex`.
        # If not, fall back to a broader pattern without '/visuals'.
        asset_main_prim_path = asset.cfg.prim_path
        pattern_with_visuals = f"{asset_main_prim_path}/{body_names_regex}/visuals"
        # Use sim_utils to check if any prims currently match this pattern
        matching_prims = sim_utils.find_matching_prim_paths(pattern_with_visuals)
        if matching_prims:
            # If matches are found, use the pattern with /visuals
            prim_path = pattern_with_visuals
        else:
            # If no matches found, fall back to the broader pattern without /visuals
            # This pattern (e.g., /World/envs/env_.*/Table/.*) should match visual prims
            # whether they end in /visuals or have other structures.
            prim_path = f"{asset_main_prim_path}/.*"
            carb.log_info(
                f"Pattern '{pattern_with_visuals}' found no prims. Falling back to '{prim_path}' for texture"
                " randomization."
            )

        # extract the replicator version
        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            texture_paths = cfg.params.get("texture_paths")
            event_name = cfg.params.get("event_name")
            texture_rotation = cfg.params.get("texture_rotation", (0.0, 0.0))

            # convert from radians to degrees
            texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)

            # Create the omni-graph node for the randomization term
            def rep_texture_randomization():
                prims_group = rep.get.prims(path_pattern=prim_path)

                with prims_group:
                    rep.randomizer.texture(
                        textures=texture_paths,
                        project_uvw=True,
                        texture_rotate=rep.distribution.uniform(*texture_rotation),
                    )
                return prims_group.node

            # Register the event to the replicator
            with rep.trigger.on_custom_event(event_name=event_name):
                rep_texture_randomization()
        else:
            # acquire stage
            stage = get_current_stage()
            prims_group = rep.functional.get.prims(path_pattern=prim_path, stage=stage)

            num_prims = len(prims_group)
            # rng that randomizes the texture and rotation
            self.texture_rng = rep.rng.ReplicatorRNG()

            # Create the material first and bind it to the prims
            for i, prim in enumerate(prims_group):
                # Disable instancble
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)

            # TODO: Should we specify the value when creating the material?
            self.material_prims = rep.functional.create_batch.material(
                mdl="OmniPBR.mdl", bind_prims=prims_group, count=num_prims, project_uvw=True
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        texture_paths: list[str],
        texture_rotation: tuple[float, float] = (0.0, 0.0),
    ):
        # note: This triggers the nodes for all the environments.
        #   We need to investigate how to make it happen only for a subset based on env_ids.
        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # extract the replicator version
        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            rep.utils.send_og_event(event_name)
        else:
            # read parameters from the configuration
            texture_paths = texture_paths if texture_paths else self._cfg.params.get("texture_paths")
            texture_rotation = (
                texture_rotation if texture_rotation else self._cfg.params.get("texture_rotation", (0.0, 0.0))
            )

            # convert from radians to degrees
            texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)

            num_prims = len(self.material_prims)
            random_textures = self.texture_rng.generator.choice(texture_paths, size=num_prims)
            random_rotations = self.texture_rng.generator.uniform(
                texture_rotation[0], texture_rotation[1], size=num_prims
            )

            # modify the material properties
            rep.functional.modify.attribute(self.material_prims, "diffuse_texture", random_textures)
            rep.functional.modify.attribute(self.material_prims, "texture_rotate", random_rotations)


class randomize_visual_color(ManagerTermBase):
    """Randomize the visual color of bodies on an asset using Replicator API.

    This function randomizes the visual color of the bodies of the asset using the Replicator API.
    The function samples random colors from the given colors and applies them to the bodies
    of the asset.

    The function assumes that the asset follows the prim naming convention as:
    "{asset_prim_path}/{mesh_name}" where the mesh name is the name of the mesh to
    which the color is applied. For instance, if the asset has a prim path "/World/asset"
    and a mesh named "body_0/mesh", the prim path for the mesh would be
    "/World/asset/body_0/mesh".

    The colors can be specified as a list of tuples of the form ``(r, g, b)`` or as a dictionary
    with the keys ``r``, ``g``, ``b`` and values as tuples of the form ``(low, high)``.
    If a dictionary is used, the function will sample random colors from the given ranges.

    .. note::
        When randomizing the color of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the randomization term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # enable replicator extension if not already enabled
        enable_extension("omni.replicator.core")
        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # read parameters from the configuration
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")
        mesh_name: str = cfg.params.get("mesh_name", "")  # type: ignore

        # check to make sure replicate_physics is set to False, else raise error
        # note: We add an explicit check here since texture randomization can happen outside of 'prestartup' mode
        #   and the event manager doesn't check in that case.
        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "Unable to randomize visual color with scene replication enabled."
                " For stable USD-level randomization, please disable scene replication"
                " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
            )

        # obtain the asset entity
        asset = env.scene[asset_cfg.name]

        # create the affected prim path
        if not mesh_name.startswith("/"):
            mesh_name = "/" + mesh_name
        mesh_prim_path = f"{asset.cfg.prim_path}{mesh_name}"
        # TODO: Need to make it work for multiple meshes.

        # extract the replicator version
        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            colors = cfg.params.get("colors")
            event_name = cfg.params.get("event_name")

            # parse the colors into replicator format
            if isinstance(colors, dict):
                # (r, g, b) - low, high --> (low_r, low_g, low_b) and (high_r, high_g, high_b)
                color_low = [colors[key][0] for key in ["r", "g", "b"]]
                color_high = [colors[key][1] for key in ["r", "g", "b"]]
                colors = rep.distribution.uniform(color_low, color_high)
            else:
                colors = list(colors)

            # Create the omni-graph node for the randomization term
            def rep_color_randomization():
                prims_group = rep.get.prims(path_pattern=mesh_prim_path)
                with prims_group:
                    rep.randomizer.color(colors=colors)

                return prims_group.node

            # Register the event to the replicator
            with rep.trigger.on_custom_event(event_name=event_name):
                rep_color_randomization()
        else:
            stage = get_current_stage()
            prims_group = rep.functional.get.prims(path_pattern=mesh_prim_path, stage=stage)

            num_prims = len(prims_group)
            self.color_rng = rep.rng.ReplicatorRNG()

            # Create the material first and bind it to the prims
            for i, prim in enumerate(prims_group):
                # Disable instancble
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)

            # TODO: Should we specify the value when creating the material?
            self.material_prims = rep.functional.create_batch.material(
                mdl="OmniPBR.mdl", bind_prims=prims_group, count=num_prims, project_uvw=True
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]],
        mesh_name: str = "",
    ):
        # note: This triggers the nodes for all the environments.
        #   We need to investigate how to make it happen only for a subset based on env_ids.

        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            rep.utils.send_og_event(event_name)
        else:
            colors = colors if colors else self._cfg.params.get("colors")

            # parse the colors into replicator format
            if isinstance(colors, dict):
                # (r, g, b) - low, high --> (low_r, low_g, low_b) and (high_r, high_g, high_b)
                color_low = [colors[key][0] for key in ["r", "g", "b"]]
                color_high = [colors[key][1] for key in ["r", "g", "b"]]
                colors = [color_low, color_high]
            else:
                colors = list(colors)

            num_prims = len(self.material_prims)
            random_colors = self.color_rng.generator.uniform(colors[0], colors[1], size=(num_prims, 3))

            rep.functional.modify.attribute(self.material_prims, "diffuse_color_constant", random_colors)


##############################################################################
# 额外的鲁棒性随机化事件（动作/传感器/故障类）
##############################################################################


class randomize_action_noise(ManagerTermBase):
    """在动作上添加噪声，模拟控制信号不完美。
    
    每个 step 在 policy 输出的动作上叠加高斯噪声或均匀噪声，
    模拟真实控制链路中的量化误差、通讯抖动等。
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        # 噪声标准差或范围
        self.noise_std = cfg.params.get("noise_std", 0.02)
        self.noise_type = cfg.params.get("noise_type", "gaussian")  # "gaussian" or "uniform"

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        noise_std: float = 0.02,
        noise_type: str = "gaussian",
    ):
        """对关节位置目标添加噪声。"""
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)
        
        # 获取当前关节位置目标
        joint_pos_target = self.asset.data.joint_pos_target[env_ids].clone()
        
        # 添加噪声
        if noise_type == "gaussian":
            noise = torch.randn_like(joint_pos_target) * noise_std
        else:  # uniform
            noise = (torch.rand_like(joint_pos_target) * 2 - 1) * noise_std
        
        joint_pos_target += noise
        
        # 写回（注意：这会在下一个 step 生效）
        self.asset.set_joint_position_target(joint_pos_target, env_ids=env_ids)


class randomize_action_delay(ManagerTermBase):
    """模拟动作延迟，使用 FIFO 缓冲区延迟命令执行。
    
    维护一个动作历史缓冲区，每次实际执行的是 delay_steps 之前的动作，
    模拟真实系统中的通讯延迟和控制周期不对齐。
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        
        # 最大延迟步数
        self.max_delay_steps = cfg.params.get("max_delay_steps", 3)
        # 为每个 env 采样一个固定的延迟步数
        self.delay_steps = torch.randint(
            0, self.max_delay_steps + 1, 
            (env.scene.num_envs,), 
            device=self.asset.device
        )
        # 动作历史缓冲区: (num_envs, max_delay_steps + 1, num_joints)
        self.action_buffer = torch.zeros(
            env.scene.num_envs,
            self.max_delay_steps + 1,
            self.asset.num_joints,
            device=self.asset.device,
        )
        self.buffer_idx = 0

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        max_delay_steps: int = 3,
    ):
        """应用延迟后的动作。"""
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)
        
        # 获取当前动作目标
        current_action = self.asset.data.joint_pos_target.clone()
        
        # 存入缓冲区
        self.action_buffer[:, self.buffer_idx] = current_action
        
        # 对每个 env，取出延迟后的动作
        delayed_actions = torch.zeros_like(current_action)
        for i in range(env.scene.num_envs):
            delay = self.delay_steps[i].item()
            delayed_idx = (self.buffer_idx - delay) % (self.max_delay_steps + 1)
            delayed_actions[i] = self.action_buffer[i, delayed_idx]
        
        # 更新缓冲区索引
        self.buffer_idx = (self.buffer_idx + 1) % (self.max_delay_steps + 1)
        
        # 应用延迟后的动作
        self.asset.set_joint_position_target(delayed_actions[env_ids], env_ids=env_ids)
        
        # reset 时重新采样延迟
        if env_ids is not None and len(env_ids) > 0:
            self.delay_steps[env_ids] = torch.randint(
                0, self.max_delay_steps + 1, (len(env_ids),), device=self.asset.device
            )
            self.action_buffer[env_ids] = 0.0


class randomize_joint_encoder_noise(ManagerTermBase):
    """关节编码器噪声和偏置随机化。
    
    模拟真实编码器的测量噪声和零点偏移：
    - 每个 env 采样一个固定的 bias（慢变化）
    - 每个 step 叠加高斯噪声（快变化）
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        
        # 噪声参数
        self.pos_noise_std = cfg.params.get("pos_noise_std", 0.01)  # rad
        self.vel_noise_std = cfg.params.get("vel_noise_std", 0.1)   # rad/s
        self.pos_bias_range = cfg.params.get("pos_bias_range", (-0.02, 0.02))  # rad
        self.vel_bias_range = cfg.params.get("vel_bias_range", (-0.05, 0.05))  # rad/s
        
        # 为每个 env 采样固定偏置
        self.pos_bias = math_utils.sample_uniform(
            self.pos_bias_range[0], self.pos_bias_range[1],
            (env.scene.num_envs, self.asset.num_joints),
            device=self.asset.device
        )
        self.vel_bias = math_utils.sample_uniform(
            self.vel_bias_range[0], self.vel_bias_range[1],
            (env.scene.num_envs, self.asset.num_joints),
            device=self.asset.device
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        pos_noise_std: float = 0.01,
        vel_noise_std: float = 0.1,
        pos_bias_range: tuple[float, float] = (-0.02, 0.02),
        vel_bias_range: tuple[float, float] = (-0.05, 0.05),
    ):
        """在关节位置和速度观测上添加噪声和偏置。
        
        注意：这个函数修改 asset.data 中的值，应该在 observation 计算之前调用。
        """
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)
        
        # 添加位置噪声和偏置
        pos_noise = torch.randn(len(env_ids), self.asset.num_joints, device=self.asset.device) * pos_noise_std
        self.asset.data.joint_pos[env_ids] += pos_noise + self.pos_bias[env_ids]
        
        # 添加速度噪声和偏置
        vel_noise = torch.randn(len(env_ids), self.asset.num_joints, device=self.asset.device) * vel_noise_std
        self.asset.data.joint_vel[env_ids] += vel_noise + self.vel_bias[env_ids]
        
        # reset 时重新采样偏置
        if env_ids is not None and len(env_ids) > 0:
            self.pos_bias[env_ids] = math_utils.sample_uniform(
                pos_bias_range[0], pos_bias_range[1],
                (len(env_ids), self.asset.num_joints),
                device=self.asset.device
            )
            self.vel_bias[env_ids] = math_utils.sample_uniform(
                vel_bias_range[0], vel_bias_range[1],
                (len(env_ids), self.asset.num_joints),
                device=self.asset.device
            )


class randomize_imu_noise_and_bias(ManagerTermBase):
    """IMU 噪声和漂移随机化。
    
    模拟真实 IMU 的测量特性：
    - 角速度白噪声 + 固定偏置
    - 线加速度白噪声 + 固定偏置
    - 可选：偏置随机游走（慢漂移）
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        
        # 噪声参数
        self.ang_vel_noise_std = cfg.params.get("ang_vel_noise_std", 0.05)  # rad/s
        self.lin_acc_noise_std = cfg.params.get("lin_acc_noise_std", 0.1)   # m/s^2
        self.ang_vel_bias_range = cfg.params.get("ang_vel_bias_range", (-0.02, 0.02))
        self.lin_acc_bias_range = cfg.params.get("lin_acc_bias_range", (-0.1, 0.1))
        self.bias_drift_std = cfg.params.get("bias_drift_std", 0.001)  # 每步漂移
        
        # 为每个 env 采样固定偏置
        self.ang_vel_bias = math_utils.sample_uniform(
            self.ang_vel_bias_range[0], self.ang_vel_bias_range[1],
            (env.scene.num_envs, 3), device=self.asset.device
        )
        self.lin_acc_bias = math_utils.sample_uniform(
            self.lin_acc_bias_range[0], self.lin_acc_bias_range[1],
            (env.scene.num_envs, 3), device=self.asset.device
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        ang_vel_noise_std: float = 0.05,
        lin_acc_noise_std: float = 0.1,
        ang_vel_bias_range: tuple[float, float] = (-0.02, 0.02),
        lin_acc_bias_range: tuple[float, float] = (-0.1, 0.1),
        bias_drift_std: float = 0.001,
    ):
        """在 IMU 观测上添加噪声和偏置。"""
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)
        
        # 偏置随机游走
        self.ang_vel_bias += torch.randn_like(self.ang_vel_bias) * bias_drift_std
        self.lin_acc_bias += torch.randn_like(self.lin_acc_bias) * bias_drift_std
        
        # 添加角速度噪声和偏置
        ang_vel_noise = torch.randn(len(env_ids), 3, device=self.asset.device) * ang_vel_noise_std
        self.asset.data.root_ang_vel_b[env_ids] += ang_vel_noise + self.ang_vel_bias[env_ids]
        
        # 添加线加速度噪声和偏置（如果有的话）
        lin_vel_noise = torch.randn(len(env_ids), 3, device=self.asset.device) * lin_acc_noise_std
        self.asset.data.root_lin_vel_b[env_ids] += lin_vel_noise + self.lin_acc_bias[env_ids]
        
        # reset 时重新采样偏置
        if env_ids is not None and len(env_ids) > 0:
            self.ang_vel_bias[env_ids] = math_utils.sample_uniform(
                ang_vel_bias_range[0], ang_vel_bias_range[1],
                (len(env_ids), 3), device=self.asset.device
            )
            self.lin_acc_bias[env_ids] = math_utils.sample_uniform(
                lin_acc_bias_range[0], lin_acc_bias_range[1],
                (len(env_ids), 3), device=self.asset.device
            )


class randomize_observation_dropout(ManagerTermBase):
    """观测丢包/传感器失效随机化。
    
    以一定概率将部分观测维度置零或保持上一帧值，
    模拟传感器偶发失效、通讯丢包等情况。
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        
        self.dropout_prob = cfg.params.get("dropout_prob", 0.01)  # 每个维度丢包概率
        self.dropout_mode = cfg.params.get("dropout_mode", "zero")  # "zero" or "hold"
        
        # 保存上一帧观测用于 hold 模式
        self.last_joint_pos = self.asset.data.joint_pos.clone()
        self.last_joint_vel = self.asset.data.joint_vel.clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        dropout_prob: float = 0.01,
        dropout_mode: str = "zero",
    ):
        """随机丢弃部分观测。"""
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)
        
        # 生成丢包掩码
        pos_dropout_mask = torch.rand(len(env_ids), self.asset.num_joints, device=self.asset.device) < dropout_prob
        vel_dropout_mask = torch.rand(len(env_ids), self.asset.num_joints, device=self.asset.device) < dropout_prob
        
        if dropout_mode == "zero":
            # 置零
            self.asset.data.joint_pos[env_ids] = torch.where(
                pos_dropout_mask, 
                torch.zeros_like(self.asset.data.joint_pos[env_ids]),
                self.asset.data.joint_pos[env_ids]
            )
            self.asset.data.joint_vel[env_ids] = torch.where(
                vel_dropout_mask,
                torch.zeros_like(self.asset.data.joint_vel[env_ids]),
                self.asset.data.joint_vel[env_ids]
            )
        else:  # hold
            # 保持上一帧值
            self.asset.data.joint_pos[env_ids] = torch.where(
                pos_dropout_mask,
                self.last_joint_pos[env_ids],
                self.asset.data.joint_pos[env_ids]
            )
            self.asset.data.joint_vel[env_ids] = torch.where(
                vel_dropout_mask,
                self.last_joint_vel[env_ids],
                self.asset.data.joint_vel[env_ids]
            )
        
        # 更新上一帧缓存
        self.last_joint_pos[env_ids] = self.asset.data.joint_pos[env_ids].clone()
        self.last_joint_vel[env_ids] = self.asset.data.joint_vel[env_ids].clone()


def randomize_constant_wind_like_force(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    force_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """施加持续的类风力/拖拽力。
    
    为每个 env 采样一个固定方向的力，整个 episode 保持不变，
    模拟持续的侧向风、线缆拉扯等恒定偏置力。
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    
    # 采样持续力（每个 env 一个固定值）
    range_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    forces = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)
    
    # 扩展到所有 body（只对 base/torso 施加）
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else 1
    forces = forces.unsqueeze(1).expand(-1, num_bodies, -1)
    torques = torch.zeros_like(forces)
    
    # 设置外力
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)


def randomize_slope_or_base_frame(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    gravity_bias_range: dict[str, tuple[float, float]],
):
    """通过修改重力方向模拟基座倾斜/坡度。
    
    在重力向量的 x, y 分量上添加小偏置，
    等效于机器人站在轻微倾斜的地面上。
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    
    # 获取当前重力
    gravity = torch.tensor(env.sim.cfg.gravity, device="cpu").unsqueeze(0)
    
    # 采样重力偏置（主要在 x, y 方向）
    range_list = [gravity_bias_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    bias = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (1, 3), device="cpu")
    
    # 应用偏置
    gravity = gravity + bias
    gravity = gravity[0].tolist()
    
    # 设置新重力
    physics_sim_view: physx.SimulationView = sim_utils.SimulationContext.instance().physics_sim_view
    physics_sim_view.set_gravity(carb.Float3(*gravity))


class randomize_joint_failure(ManagerTermBase):
    """关节故障随机化。
    
    以一定概率让某些关节"失效"：
    - 扭矩输出减半或归零
    - 关节卡死在某个位置
    模拟电机故障、驱动器失效等极端情况。
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        
        self.failure_prob = cfg.params.get("failure_prob", 0.01)  # 每个关节失效概率
        self.failure_mode = cfg.params.get("failure_mode", "weak")  # "weak", "stuck", "dead"
        self.weak_factor = cfg.params.get("weak_factor", 0.3)  # weak 模式下的扭矩衰减因子
        
        # 记录哪些关节失效
        self.failed_joints = torch.zeros(
            env.scene.num_envs, self.asset.num_joints, 
            dtype=torch.bool, device=self.asset.device
        )
        self.stuck_positions = torch.zeros(
            env.scene.num_envs, self.asset.num_joints,
            device=self.asset.device
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        failure_prob: float = 0.01,
        failure_mode: str = "weak",
        weak_factor: float = 0.3,
    ):
        """应用关节故障效果。"""
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)
        
        # 对 reset 的 env 重新采样故障状态
        new_failures = torch.rand(len(env_ids), self.asset.num_joints, device=self.asset.device) < failure_prob
        self.failed_joints[env_ids] = new_failures
        self.stuck_positions[env_ids] = self.asset.data.joint_pos[env_ids].clone()
        
        # 应用故障效果
        if failure_mode == "weak":
            # 获取当前关节扭矩限制并衰减
            # 注意：这里简化处理，实际应该修改 actuator 的 effort_limit
            pass  # 在 actuator 层面处理更合适
        elif failure_mode == "stuck":
            # 强制关节保持在失效时的位置
            stuck_mask = self.failed_joints[env_ids]
            current_pos = self.asset.data.joint_pos[env_ids].clone()
            current_pos = torch.where(stuck_mask, self.stuck_positions[env_ids], current_pos)
            self.asset.set_joint_position_target(current_pos, env_ids=env_ids)
        elif failure_mode == "dead":
            # 关节完全不响应（目标位置设为当前位置）
            dead_mask = self.failed_joints[env_ids]
            current_pos = self.asset.data.joint_pos[env_ids].clone()
            target_pos = self.asset.data.joint_pos_target[env_ids].clone()
            target_pos = torch.where(dead_mask, current_pos, target_pos)
            self.asset.set_joint_position_target(target_pos, env_ids=env_ids)


class randomize_contact_patch_slip(ManagerTermBase):
    """局部超低摩擦区域随机化。
    
    随机将部分 env 的地面/脚底摩擦设置得很低，
    模拟踩到冰面、油污等滑溜区域。
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        
        self.slip_prob = cfg.params.get("slip_prob", 0.05)  # 滑溜区域出现概率
        self.slip_friction = cfg.params.get("slip_friction", 0.1)  # 滑溜时的摩擦系数
        self.normal_friction_range = cfg.params.get("normal_friction_range", (0.6, 1.0))

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        slip_prob: float = 0.05,
        slip_friction: float = 0.1,
        normal_friction_range: tuple[float, float] = (0.6, 1.0),
    ):
        """随机设置滑溜区域。"""
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()
        
        # 获取当前材质属性
        materials = self.asset.root_physx_view.get_material_properties()
        
        # 决定哪些 env 是滑溜的
        is_slip = torch.rand(len(env_ids)) < slip_prob
        
        for i, env_id in enumerate(env_ids):
            if is_slip[i]:
                # 设置低摩擦
                materials[env_id, :, 0] = slip_friction  # static friction
                materials[env_id, :, 1] = slip_friction  # dynamic friction
            else:
                # 正常摩擦
                friction = torch.empty(materials.shape[1]).uniform_(*normal_friction_range)
                materials[env_id, :, 0] = friction
                materials[env_id, :, 1] = friction * 0.8  # dynamic < static
        
        self.asset.root_physx_view.set_material_properties(materials, env_ids)


class randomize_sensor_latency_spike(ManagerTermBase):
    """传感器延迟尖峰随机化。
    
    极少数 step 在观测上注入大延迟（使用很旧的一帧），
    模拟偶发的通讯阻塞、传感器卡顿等。
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        
        self.spike_prob = cfg.params.get("spike_prob", 0.005)  # 延迟尖峰发生概率
        self.max_latency_steps = cfg.params.get("max_latency_steps", 10)  # 最大延迟步数
        
        # 观测历史缓冲区
        self.history_buffer_pos = torch.zeros(
            env.scene.num_envs,
            self.max_latency_steps + 1,
            self.asset.num_joints,
            device=self.asset.device
        )
        self.history_buffer_vel = torch.zeros(
            env.scene.num_envs,
            self.max_latency_steps + 1,
            self.asset.num_joints,
            device=self.asset.device
        )
        self.buffer_idx = 0

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        spike_prob: float = 0.005,
        max_latency_steps: int = 10,
    ):
        """偶发注入大延迟。"""
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)
        
        # 存储当前观测
        self.history_buffer_pos[:, self.buffer_idx] = self.asset.data.joint_pos.clone()
        self.history_buffer_vel[:, self.buffer_idx] = self.asset.data.joint_vel.clone()
        
        # 决定哪些 env 发生延迟尖峰
        spike_mask = torch.rand(len(env_ids), device=self.asset.device) < spike_prob
        
        if spike_mask.any():
            # 随机选择延迟步数
            latency = torch.randint(1, max_latency_steps + 1, (len(env_ids),), device=self.asset.device)
            
            for i, env_id in enumerate(env_ids):
                if spike_mask[i]:
                    delayed_idx = (self.buffer_idx - latency[i].item()) % (self.max_latency_steps + 1)
                    self.asset.data.joint_pos[env_id] = self.history_buffer_pos[env_id, delayed_idx]
                    self.asset.data.joint_vel[env_id] = self.history_buffer_vel[env_id, delayed_idx]
        
        # 更新缓冲区索引
        self.buffer_idx = (self.buffer_idx + 1) % (self.max_latency_steps + 1)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def _validate_scale_range(
    params: tuple[float, float] | None,
    name: str,
    *,
    allow_negative: bool = False,
    allow_zero: bool = True,
) -> None:
    """
    Validates a (low, high) tuple used in scale-based randomization.

    This function ensures the tuple follows expected rules when applying a 'scale'
    operation. It performs type and value checks, optionally allowing negative or
    zero lower bounds.

    Args:
        params (tuple[float, float] | None): The (low, high) range to validate. If None,
            validation is skipped.
        name (str): The name of the parameter being validated, used for error messages.
        allow_negative (bool, optional): If True, allows the lower bound to be negative.
            Defaults to False.
        allow_zero (bool, optional): If True, allows the lower bound to be zero.
            Defaults to True.

    Raises:
        TypeError: If `params` is not a tuple of two numbers.
        ValueError: If the lower bound is negative or zero when not allowed.
        ValueError: If the upper bound is less than the lower bound.

    Example:
        _validate_scale_range((0.5, 1.5), "mass_scale")
    """
    if params is None:  # caller didn’t request randomisation for this field
        return
    low, high = params
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise TypeError(f"{name}: expected (low, high) to be a tuple of numbers, got {params}.")
    if not allow_negative and not allow_zero and low <= 0:
        raise ValueError(f"{name}: lower bound must be > 0 when using the 'scale' operation (got {low}).")
    if not allow_negative and allow_zero and low < 0:
        raise ValueError(f"{name}: lower bound must be ≥ 0 when using the 'scale' operation (got {low}).")
    if high < low:
        raise ValueError(f"{name}: upper bound ({high}) must be ≥ lower bound ({low}).")
