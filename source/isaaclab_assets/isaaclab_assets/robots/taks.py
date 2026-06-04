"""Configuration for Taks_T1 robot.

The following configurations are available:

* :obj:`TAKS_T1_CFG`: Full-body Taks_T1 humanoid robot (legs + waist + dual arms + neck) for locomotion tasks
* :obj:`SEMI_TAKS_T1_CFG`: Semi-body Taks_T1 robot (waist + dual arms + neck, fixed base)

"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Full-body Taks_T1 USD路径
_TAKS_T1_USD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "taks", "Taks_T1.usd"
)

# Semi-body Taks_T1 USD路径
_SEMI_TAKS_T1_USD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "taks", "Semi_Taks_T1.usd"
)

##
# Full-body Configuration
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
    # 10Hz, 阻尼比2.0
    actuators={
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
                ".*_hip_yaw_joint": 25.0,
                ".*_hip_roll_joint": 25.0,
            },
            stiffness=589.409607,
            damping=37.522984,
            armature=0.149299,
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
                ".*_hip_pitch_joint": 25.0,
                ".*_knee_joint": 25.0,
            },
            stiffness=219.499985,
            damping=13.973804,
            armature=0.055600,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            stiffness=112.434517,
            damping=7.157804,
            armature=0.028480,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=97.0,
            velocity_limit_sim=25.0,
            stiffness=589.409607,
            damping=37.522984,
            armature=0.149299,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            stiffness=112.434517,
            damping=7.157804,
            armature=0.028480,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=7.0,
            velocity_limit_sim=8.0,
            stiffness=6.632374,
            damping=0.422230,
            armature=0.001680,
        ),
        "neck": ImplicitActuatorCfg(
            joint_names_expr=[
                "neck_yaw_joint",
                "neck_roll_joint",
                "neck_pitch_joint",
            ],
            effort_limit_sim=3.0,
            velocity_limit_sim=8.0,
            stiffness=4.936697,
            damping=0.157140,
            armature=0.000313,
        ),
    },
)
"""Configuration for the full-body Taks_T1 humanoid robot for locomotion tasks.

Joint structure:
- Legs: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll (6 DOF per leg)
- Waist: waist_yaw, waist_roll, waist_pitch (3 DOF)
- Arms: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw (7 DOF per arm)
- Neck: neck_yaw, neck_roll, neck_pitch (3 DOF)
Total: 32 DOF
"""

##
# Semi-body Configuration
##

SEMI_TAKS_T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_SEMI_TAKS_T1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            ".*_shoulder_pitch_joint": 0.16,
            ".*_elbow_joint": 1.10,
            "waist_.*": 0.0,
            "neck_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    # 10Hz, 阻尼比2.0
    actuators={
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=97.0,
            velocity_limit_sim=4.19,
            stiffness=589.409607,
            damping=37.522984,
            armature=0.149299,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "left_elbow_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_joint",
            ],
            effort_limit_sim=27.0,
            velocity_limit_sim=5.4454,
            stiffness=112.434517,
            damping=7.157804,
            armature=0.028480,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint",
                "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
            ],
            effort_limit_sim=9.0,
            velocity_limit_sim=20.944,
            stiffness=6.632374,
            damping=0.422230,
            armature=0.001680,
        ),
        "neck": ImplicitActuatorCfg(
            joint_names_expr=[
                "neck_yaw_joint",
                "neck_roll_joint",
                "neck_pitch_joint",
            ],
            effort_limit_sim=3.0,
            velocity_limit_sim=15.71,
            stiffness=1.234174,
            damping=0.078570,
            armature=0.000313,
        ),
    },
)
"""Semi-body Taks_T1 robot configuration for manipulation tasks.

Fixed base with waist + dual arms + neck, no legs.
Joint structure:
- Waist: waist_yaw, waist_roll, waist_pitch (3 DOF)
- Arms: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow (4 DOF per arm)
- Wrists: wrist_roll, wrist_yaw, wrist_pitch (3 DOF per arm)
- Neck: neck_yaw, neck_roll, neck_pitch (3 DOF)
Total: 20 DOF
"""