"""Configuration for Taks_T1 robot.

The following configurations are available:

* :obj:`TAKS_T1_CFG`: Taks_T1 humanoid robot configured for locomotion tasks

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

TAKS_T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/isaaclab_assets/isaaclab_assets/robots/Taks_T1/Taks_T1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        rot=(0.7071, 0, 0, 0.7071),
        joint_pos={
            ".*_elbow_joint": 1.56,  # 肘关节初始位置设为自然下垂状态
            "waist_pitch_joint": 0.08,
            "left_shoulder_roll_joint": 0.20,
            "right_shoulder_roll_joint": -0.20,
            "left_knee_joint": 0.57,
            "right_knee_joint": 0.57,
            "left_hip_pitch_joint": -0.57,
            "right_hip_pitch_joint": -0.57,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    actuators={
        # 腿部关节配置 - 扭矩来自URDF/XML
        "legs": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 97.0,
                ".*_hip_roll_joint": 97.0,
                ".*_hip_pitch_joint": 120.0,
                ".*_knee_joint": 120.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 32.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.03,
                ".*_knee_joint": 0.03,
            },
            saturation_effort=100.0,
        ),
        # 脚踝关节配置 - 扭矩来自URDF/XML: 27 Nm
        "feet": DCMotorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=27.0,
            velocity_limit=37.0,
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.2,
                ".*_ankle_roll_joint": 0.1,
            },
            armature=0.03,
            saturation_effort=27.0,
        ),
        # 腰部关节配置 - 扭矩来自URDF/XML: 97 Nm
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=97,
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
        # 手臂关节配置 - 扭矩来自URDF/XML: 27 Nm
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=27,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_joint": 0.01,
            },
        ),
        # 手腕关节配置 - 扭矩来自URDF/XML: 7 Nm
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=7,
            stiffness=20.0,
            damping=5.0,
            armature=0.01,
        ),
        # 颈部关节配置 - 扭矩来自URDF/XML: 3 Nm
        "neck": ImplicitActuatorCfg(
            joint_names_expr=[
                "neck_yaw_joint",
                "neck_roll_joint",
                "neck_pitch_joint",
            ],
            effort_limit_sim=3,
            stiffness=5.0,
            damping=1.0,
            armature=0.01,
        ),
    },
)
"""Configuration for the Taks_T1 Humanoid robot for locomotion tasks.

This configuration sets up the Taks_T1 humanoid robot for locomotion tasks,
allowing both locomotion and manipulation capabilities. The robot can be configured
for either fixed base or mobile scenarios by modifying the fix_root_link parameter.

Key features:
- Configurable base (fixed or mobile) via fix_root_link parameter
- Optimized actuator parameters for locomotion tasks
- Enhanced arm and neck configurations

Joint structure based on URDF:
- Legs: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll (6 DOF per leg)
- Waist: waist_yaw, waist_roll, waist_pitch (3 DOF)
- Arms: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw (7 DOF per arm)
- Neck: neck_yaw, neck_roll, neck_pitch (3 DOF)
Total: 35 DOF
"""
