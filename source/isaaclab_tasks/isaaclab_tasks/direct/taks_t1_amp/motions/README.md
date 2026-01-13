# Taks_T1 Motion Files

Motion files for Taks_T1 AMP environment in NumPy-file format.

## Data Format

| Key | Dtype | Shape | Description |
| --- | ---- | ----- | ----------- |
| `fps` | int64 | () | FPS at which motion was sampled |
| `dof_names` | unicode string | (35,) | Skeleton DOF names |
| `body_names` | unicode string | (33,) | Skeleton body names |
| `dof_positions` | float32 | (N, 35) | Skeleton DOF positions |
| `dof_velocities` | float32 | (N, 35) | Skeleton DOF velocities |
| `body_positions` | float32 | (N, 33, 3) | Skeleton body positions |
| `body_rotations` | float32 | (N, 33, 4) | Skeleton body rotations (as `wxyz` quaternion) |
| `body_linear_velocities` | float32 | (N, 33, 3) | Skeleton body linear velocities |
| `body_angular_velocities` | float32 | (N, 33, 3) | Skeleton body angular velocities |

## Taks_T1 Joint Structure (35 DOF)

- **Legs (12 DOF)**: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll (6 per leg)
- **Waist (3 DOF)**: waist_yaw, waist_roll, waist_pitch
- **Arms (14 DOF)**: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_yaw, wrist_pitch (7 per arm)
- **Neck (3 DOF)**: neck_yaw, neck_roll, neck_pitch

## Motion Conversion

Use `convert_motion.py` to create or convert motion files:

```bash
# Create default walk motion
python convert_motion.py --output taks_t1_walk.npz --type walk --frames 600 --fps 60

# Create default run motion
python convert_motion.py --output taks_t1_run.npz --type run --frames 300 --fps 60

# Convert from other format (requires retargeting implementation)
python convert_motion.py --input source_motion.npz --output taks_t1_motion.npz --fps 60
```

## Key Bodies for AMP

The following bodies are used as key bodies for AMP observation:
- `left_wrist_pitch_link` (left hand)
- `right_wrist_pitch_link` (right hand)
- `left_ankle_roll_link` (left foot)
- `right_ankle_roll_link` (right foot)
