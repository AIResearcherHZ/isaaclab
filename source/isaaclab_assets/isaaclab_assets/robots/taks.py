"""Configuration for Taks_T1 robot.

The following configurations are available:

* :obj:`TAKS_T1_CFG`: Taks_T1 humanoid robot configured for locomotion tasks

"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Get the absolute path to the Taks_T1 USD file
_TAKS_T1_USD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Taks_T1", "Taks_T1.usd"
)

##
# Configuration
##

TAKS_T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_TAKS_T1_USD_PATH,
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
        pos=(0.0, 0.0, 0.75),
        rot=(0.7071, 0, 0, 0.7071),
        joint_pos={
            "left_shoulder_roll_joint": 0.16,
            "right_shoulder_roll_joint": -0.16,
            ".*_shoulder_pitch_joint": 0.16,
            ".*_elbow_joint": 1.10,
            ".*_hip_pitch_joint": -0.14,
            ".*_knee_joint": 0.36,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    actuators={
        # 腿部关节配置 - 扭矩来自URDF/XML
        "legs_hip_yaw_roll": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 97.0,
                ".*_hip_roll_joint": 97.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 4.19,
                ".*_hip_roll_joint": 4.19,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
            },
        ),
        "legs_hip_pitch_knee": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 120.0,
                ".*_knee_joint": 120.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 7.33,
                ".*_knee_joint": 7.33,
            },
            stiffness={
                ".*_hip_pitch_joint": 150.0,
                ".*_knee_joint": 150.0,
            },
            damping={
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
        ),
        # 脚踝关节配置 - 扭矩来自URDF/XML: 峰值 27 Nm，额定 9 Nm
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=27.0,
            velocity_limit_sim=5.4454,
            stiffness={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 40.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 2.0,
            },
            armature=0.01,
        ),
        # 腰部关节配置 - 扭矩来自URDF/XML: 峰值 97 Nm，额定 30 Nm
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=97,
            velocity_limit_sim=4.19,
            stiffness=150.0,
            damping=5.0,
            armature=0.01,
        ),
        # 手臂关节配置 - 扭矩来自URDF/XML: 峰值 27 Nm，额定 9 Nm
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=9,
            velocity_limit_sim=5.4454,
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_joint": 40.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 5.0,
                ".*_shoulder_roll_joint": 5.0,
                ".*_shoulder_yaw_joint": 5.0,
                ".*_elbow_joint": 5.0,
            },
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_joint": 0.01,
            },
        ),
        # 手腕关节配置 - 扭矩来自URDF/XML: 峰值 7 Nm，额定 3 Nm
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=3,
            velocity_limit_sim=20.944,
            stiffness=20.0,
            damping=2.0,
            armature=0.001,
        ),
        # 颈部关节配置 - 扭矩来自URDF/XML: 峰值 3 Nm，额定 0.8 Nm
        "neck": ImplicitActuatorCfg(
            joint_names_expr=[
                "neck_yaw_joint",
                "neck_roll_joint",
                "neck_pitch_joint",
            ],
            effort_limit_sim=0.8,
            velocity_limit_sim=15.71,
            stiffness=10.0,
            damping=2.0,
            armature=0.001,
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
