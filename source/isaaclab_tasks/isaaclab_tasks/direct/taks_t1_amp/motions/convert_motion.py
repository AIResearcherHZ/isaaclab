#!/usr/bin/env python3
"""Taks_T1 Motion数据转换脚本 - 从humanoid_amp格式转换到Taks_T1格式"""

import argparse
import numpy as np

# Taks_T1的32个DOF名称（训练顺序，与S2S2R/IO描述一致）
TAKS_T1_DOF_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "neck_yaw_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint", "neck_roll_joint",
    "right_shoulder_roll_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "neck_pitch_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
]

# Taks_T1的33个body名称
TAKS_T1_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_yaw_link", "left_wrist_pitch_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_yaw_link", "right_wrist_pitch_link",
    "neck_yaw_link", "neck_roll_link", "neck_pitch_link",
]

# humanoid_28 DOF到Taks_T1 DOF的映射（训练顺序索引）
HUMANOID_TO_TAKS_DOF = {
    "left_hip_y": 0, "right_hip_y": 1,    # hip_pitch
    "left_hip_x": 3, "right_hip_x": 4,    # hip_roll
    "left_hip_z": 6, "right_hip_z": 7,    # hip_yaw
    "left_knee": 9, "right_knee": 10,     # knee
    "left_ankle_y": 14, "right_ankle_y": 15,  # ankle_pitch
    "left_ankle_x": 19, "right_ankle_x": 20,  # ankle_roll
    "abdomen_z": 2, "abdomen_x": 5, "abdomen_y": 8,  # waist
    "left_shoulder_y": 11, "right_shoulder_y": 13,   # shoulder_pitch
    "left_shoulder_x": 16, "right_shoulder_x": 18,   # shoulder_roll
    "left_shoulder_z": 21, "right_shoulder_z": 23,   # shoulder_yaw
    "left_elbow": 24, "right_elbow": 25,  # elbow
    "neck_z": 12, "neck_x": 17, "neck_y": 22,  # neck
}

# URDF关节偏移量（从Taks_T1.urdf提取）
JOINT_OFFSETS = {
    "left_hip_pitch": [0.0, 0.0919, -0.08747], "left_hip_roll": [0.00795, 0.0765, -0.04167],
    "left_hip_yaw": [0.00829, 0.0, -0.1094], "left_knee": [-0.0484, 0.0, -0.1665],
    "left_ankle_pitch": [0.0, 0.0227, -0.2985], "left_ankle_roll": [0.0, -0.024, 0.0],
    "right_hip_pitch": [0.0, -0.0913, -0.08747], "right_hip_roll": [0.00795, -0.077, -0.04167],
    "right_hip_yaw": [0.00829, 0.0, -0.1094], "right_knee": [-0.0484, -0.0005, -0.1665],
    "right_ankle_pitch": [0.0, -0.0222, -0.2985], "right_ankle_roll": [0.0, 0.024, 0.0],
    "waist_yaw": [0.0, 0.0, 0.0], "waist_roll": [0.03813, 3.7e-05, 0.050906],
    "waist_pitch": [-0.03813, -0.032925, 0.079963],
    "left_shoulder_pitch": [0.0, 0.13582, 0.33654], "left_shoulder_roll": [0.03476, 0.0835, 0.000425],
    "left_shoulder_yaw": [-0.034375, 0.0, -0.12], "left_elbow": [0.000364, 0.029113, -0.18099],
    "left_wrist_roll": [0.13, -0.029125, 0.0], "left_wrist_yaw": [0.121, 0.0, -0.0333],
    "left_wrist_pitch": [0.0, 0.022502, 0.033299],
    "right_shoulder_pitch": [0.0, -0.069974, 0.33654], "right_shoulder_roll": [0.02776, -0.0835, 0.000425],
    "right_shoulder_yaw": [-0.027375, 0.0, -0.12], "right_elbow": [0.000364, 0.027375, -0.18099],
    "right_wrist_roll": [0.13, -0.027375, 0.0], "right_wrist_yaw": [0.121, 0.0, -0.0333],
    "right_wrist_pitch": [0.0, -0.0225, 0.0333],
    "neck_yaw": [0.0, 0.032906, 0.37165], "neck_roll": [-0.0719, 0.0, 0.0708],
    "neck_pitch": [0.0719, 0.02095, 0.0],
}

# 转换为numpy数组
for k, v in JOINT_OFFSETS.items():
    JOINT_OFFSETS[k] = np.array(v, dtype=np.float32)


def quat_to_rot(q):
    """四元数(wxyz)转旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ], dtype=np.float32)


def rot_to_quat(R):
    """旋转矩阵转四元数(wxyz)"""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def rot_x(a): 
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)

def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)


def slerp(q0, q1, t):
    """四元数球面插值"""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return q0 + t * (q1 - q0)
    theta = np.arccos(np.clip(dot, -1, 1))
    sin_theta = np.sin(theta)
    return (np.sin((1-t)*theta)/sin_theta) * q0 + (np.sin(t*theta)/sin_theta) * q1


def compute_fk(pelvis_pos, pelvis_quat, joint_pos):
    """正运动学：计算所有body的位置和旋转"""
    num_bodies = len(TAKS_T1_BODY_NAMES)
    positions = np.zeros((num_bodies, 3), dtype=np.float32)
    rotations = np.zeros((num_bodies, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    
    positions[0] = pelvis_pos
    rotations[0] = pelvis_quat
    pelvis_R = quat_to_rot(pelvis_quat)
    
    # 定义运动链: (body_idx, parent_idx, joint_name, axis)
    chains = [
        # 左腿
        (1, 0, "left_hip_pitch", "y"), (2, 1, "left_hip_roll", "x"), (3, 2, "left_hip_yaw", "z"),
        (4, 3, "left_knee", "y"), (5, 4, "left_ankle_pitch", "y"), (6, 5, "left_ankle_roll", "x"),
        # 右腿
        (7, 0, "right_hip_pitch", "y"), (8, 7, "right_hip_roll", "x"), (9, 8, "right_hip_yaw", "z"),
        (10, 9, "right_knee", "y"), (11, 10, "right_ankle_pitch", "y"), (12, 11, "right_ankle_roll", "x"),
        # 腰部
        (13, 0, "waist_yaw", "z"), (14, 13, "waist_roll", "x"), (15, 14, "waist_pitch", "y"),
        # 左臂
        (16, 15, "left_shoulder_pitch", "y"), (17, 16, "left_shoulder_roll", "x"), (18, 17, "left_shoulder_yaw", "z"),
        (19, 18, "left_elbow", "y"), (20, 19, "left_wrist_roll", "x"), (21, 20, "left_wrist_yaw", "z"), (22, 21, "left_wrist_pitch", "y"),
        # 右臂
        (23, 15, "right_shoulder_pitch", "y"), (24, 23, "right_shoulder_roll", "x"), (25, 24, "right_shoulder_yaw", "z"),
        (26, 25, "right_elbow", "y"), (27, 26, "right_wrist_roll", "x"), (28, 27, "right_wrist_yaw", "z"), (29, 28, "right_wrist_pitch", "y"),
        # 颈部
        (30, 15, "neck_yaw", "z"), (31, 30, "neck_roll", "x"), (32, 31, "neck_pitch", "y"),
    ]
    
    # joint_pos索引映射
    joint_idx = {
        "left_hip_pitch": 0, "left_hip_roll": 1, "left_hip_yaw": 2, "left_knee": 3, "left_ankle_pitch": 4, "left_ankle_roll": 5,
        "right_hip_pitch": 6, "right_hip_roll": 7, "right_hip_yaw": 8, "right_knee": 9, "right_ankle_pitch": 10, "right_ankle_roll": 11,
        "waist_yaw": 12, "waist_roll": 13, "waist_pitch": 14,
        "left_shoulder_pitch": 15, "left_shoulder_roll": 16, "left_shoulder_yaw": 17, "left_elbow": 18,
        "left_wrist_roll": 19, "left_wrist_yaw": 20, "left_wrist_pitch": 21,
        "right_shoulder_pitch": 22, "right_shoulder_roll": 23, "right_shoulder_yaw": 24, "right_elbow": 25,
        "right_wrist_roll": 26, "right_wrist_yaw": 27, "right_wrist_pitch": 28,
        "neck_yaw": 29, "neck_roll": 30, "neck_pitch": 31,
    }
    
    rot_funcs = {"x": rot_x, "y": rot_y, "z": rot_z}
    parent_rots = {0: pelvis_R}
    
    for body_i, parent_i, joint_name, axis in chains:
        parent_R = parent_rots[parent_i]
        parent_pos = positions[parent_i]
        angle = joint_pos[joint_idx[joint_name]] if joint_idx[joint_name] < len(joint_pos) else 0.0
        
        offset = JOINT_OFFSETS[joint_name]
        pos = parent_pos + parent_R @ offset
        R = parent_R @ rot_funcs[axis](angle)
        
        positions[body_i] = pos
        rotations[body_i] = rot_to_quat(R)
        parent_rots[body_i] = R
    
    return positions, rotations


def convert_from_humanoid(input_file, output_file):
    """从humanoid_amp格式转换到Taks_T1格式，保留pelvis旋转"""
    data = np.load(input_file)
    
    src_fps = int(data["fps"])
    num_frames = data["dof_positions"].shape[0]
    src_dof_names = [str(n) for n in data["dof_names"]]
    src_dof_idx = {name: i for i, name in enumerate(src_dof_names)}
    
    print(f"Converting: {input_file}")
    print(f"  Frames: {num_frames}, FPS: {src_fps}")
    
    num_dofs = len(TAKS_T1_DOF_NAMES)
    num_bodies = len(TAKS_T1_BODY_NAMES)
    dt = 1.0 / src_fps
    
    # 初始化输出数据
    dof_positions = np.zeros((num_frames, num_dofs), dtype=np.float32)
    dof_velocities = np.zeros((num_frames, num_dofs), dtype=np.float32)
    body_positions = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_rotations = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_linear_velocities = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_angular_velocities = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    
    # 获取源数据
    src_dof_pos = data["dof_positions"]
    src_body_pos = data["body_positions"]
    src_body_rot = data["body_rotations"]
    src_body_ang_vel = data["body_angular_velocities"]
    
    for i in range(num_frames):
        # 映射DOF
        for src_name, tgt_idx in HUMANOID_TO_TAKS_DOF.items():
            if src_name in src_dof_idx:
                dof_positions[i, tgt_idx] = src_dof_pos[i, src_dof_idx[src_name]]
        
        # 获取pelvis位置和旋转（直接从源数据复制）
        pelvis_pos = src_body_pos[i, 0].astype(np.float32)
        pelvis_quat = src_body_rot[i, 0].astype(np.float32)
        
        # 使用正运动学计算所有body位置和旋转
        body_positions[i], body_rotations[i] = compute_fk(pelvis_pos, pelvis_quat, dof_positions[i])
        
        # 复制pelvis角速度
        body_angular_velocities[i, 0] = src_body_ang_vel[i, 0]
    
    # 计算速度（有限差分）
    for i in range(1, num_frames):
        dof_velocities[i] = (dof_positions[i] - dof_positions[i-1]) / dt
        body_linear_velocities[i] = (body_positions[i] - body_positions[i-1]) / dt
        # 角速度用有限差分近似（除了pelvis已复制）
        for b in range(1, num_bodies):
            dq = body_rotations[i, b] - body_rotations[i-1, b]
            body_angular_velocities[i, b] = 2 * dq[1:4] / dt
    
    motion_data = {
        "fps": np.int64(src_fps),
        "dof_names": np.array(TAKS_T1_DOF_NAMES, dtype="U50"),
        "body_names": np.array(TAKS_T1_BODY_NAMES, dtype="U50"),
        "dof_positions": dof_positions,
        "dof_velocities": dof_velocities,
        "body_positions": body_positions,
        "body_rotations": body_rotations,
        "body_linear_velocities": body_linear_velocities,
        "body_angular_velocities": body_angular_velocities,
    }
    
    np.savez(output_file, **motion_data)
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert humanoid_amp motion to Taks_T1 format")
    parser.add_argument("--input", type=str, required=True, help="Input humanoid_amp npz file")
    parser.add_argument("--output", type=str, required=True, help="Output Taks_T1 npz file")
    args = parser.parse_args()
    
    convert_from_humanoid(args.input, args.output)


if __name__ == "__main__":
    main()
