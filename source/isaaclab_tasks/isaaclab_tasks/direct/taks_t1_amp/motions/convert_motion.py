#!/usr/bin/env python3
"""Taks_T1 Motion数据转换脚本 - 从 humanoid_amp格式转换到Taks_T1格式

方法：直接映射关节角度和body数据，不做FK重计算。
humanoid和Taks_T1骨骼结构不同，但关节角度语义相同。

humanoid_amp关节命名约定: _x=roll, _y=pitch, _z=yaw
Taks_T1关节命名约定: _roll, _pitch, _yaw
"""

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

# 训练顺序索引映射
TRAINING_IDX = {name: i for i, name in enumerate(TAKS_T1_DOF_NAMES)}

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

# humanoid DOF到Taks_T1 DOF的映射（humanoid关节名 -> taks_t1关节名）
HUMANOID_TO_TAKS_DOF = {
    "left_hip_y": "left_hip_pitch_joint",
    "left_hip_x": "left_hip_roll_joint",
    "left_hip_z": "left_hip_yaw_joint",
    "right_hip_y": "right_hip_pitch_joint",
    "right_hip_x": "right_hip_roll_joint",
    "right_hip_z": "right_hip_yaw_joint",
    "left_knee": "left_knee_joint",
    "right_knee": "right_knee_joint",
    "left_ankle_y": "left_ankle_pitch_joint",
    "left_ankle_x": "left_ankle_roll_joint",
    "right_ankle_y": "right_ankle_pitch_joint",
    "right_ankle_x": "right_ankle_roll_joint",
    "abdomen_z": "waist_yaw_joint",
    "abdomen_x": "waist_roll_joint",
    "abdomen_y": "waist_pitch_joint",
    "left_shoulder_y": "left_shoulder_pitch_joint",
    "left_shoulder_x": "left_shoulder_roll_joint",
    "left_shoulder_z": "left_shoulder_yaw_joint",
    "right_shoulder_y": "right_shoulder_pitch_joint",
    "right_shoulder_x": "right_shoulder_roll_joint",
    "right_shoulder_z": "right_shoulder_yaw_joint",
    "left_elbow": "left_elbow_joint",
    "right_elbow": "right_elbow_joint",
    "neck_z": "neck_yaw_joint",
    "neck_x": "neck_roll_joint",
    "neck_y": "neck_pitch_joint",
}

# humanoid body到Taks_T1 body的映射（用于复制body数据）
# humanoid: pelvis, torso, head, right_upper_arm, right_lower_arm, right_hand,
#           left_upper_arm, left_lower_arm, left_hand, right_thigh, right_shin, right_foot,
#           left_thigh, left_shin, left_foot
HUMANOID_TO_TAKS_BODY = {
    "pelvis": "pelvis",
    "torso": "torso_link",
    "head": "neck_pitch_link",
    "right_upper_arm": "right_shoulder_yaw_link",
    "right_lower_arm": "right_elbow_link",
    "right_hand": "right_wrist_pitch_link",
    "left_upper_arm": "left_shoulder_yaw_link",
    "left_lower_arm": "left_elbow_link",
    "left_hand": "left_wrist_pitch_link",
    "right_thigh": "right_hip_yaw_link",
    "right_shin": "right_knee_link",
    "right_foot": "right_ankle_roll_link",
    "left_thigh": "left_hip_yaw_link",
    "left_shin": "left_knee_link",
    "left_foot": "left_ankle_roll_link",
}


def convert_from_humanoid(input_file, output_file):
    """从humanoid_amp格式转换到Taks_T1格式
    
    方法：直接映射关节角度和body数据，不做FK重计算。
    body数据从humanoid直接映射到对应的Taks_T1 body。
    """
    data = np.load(input_file)
    
    src_fps = int(data["fps"])
    num_frames = data["dof_positions"].shape[0]
    src_dof_names = [str(n) for n in data["dof_names"]]
    src_dof_idx = {name: i for i, name in enumerate(src_dof_names)}
    src_body_names = [str(n) for n in data["body_names"]]
    src_body_idx = {name: i for i, name in enumerate(src_body_names)}
    
    print(f"Converting: {input_file}")
    print(f"  Frames: {num_frames}, FPS: {src_fps}")
    
    num_dofs = len(TAKS_T1_DOF_NAMES)
    num_bodies = len(TAKS_T1_BODY_NAMES)
    tgt_body_idx = {name: i for i, name in enumerate(TAKS_T1_BODY_NAMES)}
    dt = 1.0 / src_fps
    
    # 初始化输出数据
    dof_positions = np.zeros((num_frames, num_dofs), dtype=np.float32)
    dof_velocities = np.zeros((num_frames, num_dofs), dtype=np.float32)
    body_positions = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_rotations = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_rotations[:, :, 0] = 1.0  # 默认单位四元数
    body_linear_velocities = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_angular_velocities = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    
    # 获取源数据
    src_dof_pos = data["dof_positions"]
    src_dof_vel = data["dof_velocities"]
    src_body_pos = data["body_positions"]
    src_body_rot = data["body_rotations"]
    src_body_lin_vel = data["body_linear_velocities"]
    src_body_ang_vel = data["body_angular_velocities"]
    
    # 映射DOF: humanoid -> taks_t1 (训练顺序)
    for src_name, tgt_name in HUMANOID_TO_TAKS_DOF.items():
        if src_name in src_dof_idx:
            tgt_idx = TRAINING_IDX[tgt_name]
            src_idx = src_dof_idx[src_name]
            dof_positions[:, tgt_idx] = src_dof_pos[:, src_idx]
            dof_velocities[:, tgt_idx] = src_dof_vel[:, src_idx]
    
    # 映射body数据: humanoid -> taks_t1
    for src_name, tgt_name in HUMANOID_TO_TAKS_BODY.items():
        if src_name in src_body_idx and tgt_name in tgt_body_idx:
            src_idx = src_body_idx[src_name]
            tgt_idx = tgt_body_idx[tgt_name]
            body_positions[:, tgt_idx] = src_body_pos[:, src_idx]
            body_rotations[:, tgt_idx] = src_body_rot[:, src_idx]
            body_linear_velocities[:, tgt_idx] = src_body_lin_vel[:, src_idx]
            body_angular_velocities[:, tgt_idx] = src_body_ang_vel[:, src_idx]
    
    # 对于没有映射的body，使用链上最近的已映射body进行插值
    # 定义每个body应该参考的body
    body_ref_map = {
        # 左腿链: pelvis -> hip_pitch -> hip_roll -> hip_yaw(thigh) -> knee(shin) -> ankle_pitch -> ankle_roll(foot)
        "left_hip_pitch_link": "pelvis",
        "left_hip_roll_link": "pelvis", 
        "left_ankle_pitch_link": "left_knee_link",  # 在knee和foot之间插值
        # 右腿链
        "right_hip_pitch_link": "pelvis",
        "right_hip_roll_link": "pelvis",
        "right_ankle_pitch_link": "right_knee_link",
        # 腰部链: pelvis -> waist_yaw -> waist_roll -> torso
        "waist_yaw_link": "pelvis",
        "waist_roll_link": "pelvis",
        # 左臂链: torso -> shoulder_pitch -> shoulder_roll -> shoulder_yaw(upper_arm) -> elbow(lower_arm) -> wrist_roll -> wrist_yaw -> wrist_pitch(hand)
        "left_shoulder_pitch_link": "torso_link",
        "left_shoulder_roll_link": "torso_link",
        "left_wrist_roll_link": "left_elbow_link",
        "left_wrist_yaw_link": "left_elbow_link",
        # 右臂链
        "right_shoulder_pitch_link": "torso_link",
        "right_shoulder_roll_link": "torso_link",
        "right_wrist_roll_link": "right_elbow_link",
        "right_wrist_yaw_link": "right_elbow_link",
        # 颈部链: torso -> neck_yaw -> neck_roll -> neck_pitch(head)
        "neck_yaw_link": "torso_link",
        "neck_roll_link": "torso_link",
    }
    
    for name, ref_name in body_ref_map.items():
        if name in tgt_body_idx and ref_name in tgt_body_idx:
            i = tgt_body_idx[name]
            ref_idx = tgt_body_idx[ref_name]
            # 检查是否未被映射（仍是默认值）
            if np.all(body_rotations[:, i, 0] == 1.0):
                body_positions[:, i] = body_positions[:, ref_idx]
                body_rotations[:, i] = body_rotations[:, ref_idx]
                body_linear_velocities[:, i] = body_linear_velocities[:, ref_idx]
                body_angular_velocities[:, i] = body_angular_velocities[:, ref_idx]
    
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
