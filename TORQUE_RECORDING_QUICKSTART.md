# 扭矩记录功能 - 快速开始

## 最简单的使用方式

### 训练时记录扭矩

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-A1-v0 \
    --enable_torque_recording
```

### 推理时记录扭矩

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-A1-Play-v0 \
    --enable_torque_recording
```

## 操作步骤

1. **启动程序**：运行上述命令
2. **开始录制**：按 `,` 键
3. **停止录制**：按 `.` 键
4. **查看结果**：检查 `torque_logs/` 目录

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enable_torque_recording` | 启用扭矩记录 | False |
| `--torque_recording_dir` | 保存目录 | `torque_logs` |
| `--torque_recording_env_id` | 记录的环境ID | 0 |

## 完整示例

```bash
# 自定义保存目录和环境ID
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-G1-v0 \
    --enable_torque_recording \
    --torque_recording_dir logs/g1_torques \
    --torque_recording_env_id 0 \
    --num_envs 16
```

## 输出文件

每次录制会生成两个文件：

- `torque_env{id}_{timestamp}.npz` - 原始数据
- `torque_env{id}_{timestamp}.png` - 可视化图表

## 数据读取示例

```python
import numpy as np

# 加载数据
data = np.load('torque_logs/torque_env0_20231122_143025.npz')

# 访问数据
timestamps = data['timestamps']    # 时间序列
torques = data['torques']          # 扭矩数据 (steps, joints)
joint_names = data['joint_names']  # 关节名称列表

# 示例：打印第一个关节的扭矩统计
print(f"关节: {joint_names[0]}")
print(f"平均扭矩: {torques[:, 0].mean():.2f} N·m")
print(f"最大扭矩: {torques[:, 0].max():.2f} N·m")
```

## 依赖安装

```bash
pip install pynput matplotlib
```

## 常见问题

**Q: 按键没有反应？**
A: 确保终端窗口有焦点，或在图形界面下运行

**Q: 找不到robot？**
A: 确保环境配置中有 `scene.robot` 对象

**Q: 想录制多个环境？**
A: 目前只支持单个环境，可通过 `--torque_recording_env_id` 选择不同环境

---

更多详细信息请查看 `TORQUE_RECORDING_USAGE.md`
