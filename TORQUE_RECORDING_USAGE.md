# 扭矩记录功能使用说明

## 功能概述

扭矩记录功能允许您在训练和推理过程中实时记录机器人关节的扭矩数据，并自动生成可视化曲线图。

## 主要特性

- **键盘控制**：使用 `,` 键开始录制，`.` 键停止录制并保存
- **自适应关节名称**：自动获取环境中的关节名称，无需手动配置
- **可视化**：自动生成扭矩曲线图，包含统计信息（均值、最大值、最小值）
- **数据保存**：保存原始数据（.npz格式）和图形（.png格式）

## 使用方法

### 方法1：通过命令行参数启用（推荐）

#### 训练时记录
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-A1-v0 \
    --enable_torque_recording
```

#### 推理时记录
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-A1-Play-v0 \
    --enable_torque_recording
```

#### 自定义保存目录和环境ID
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-A1-v0 \
    --enable_torque_recording \
    --torque_recording_dir logs/my_torque_data \
    --torque_recording_env_id 0
```

### 方法2：在配置文件中启用

在您的agent配置文件中（例如 `rsl_rl_ppo_cfg.py`）添加以下参数：

```python
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

@configclass
class YourAgentCfg(RslRlOnPolicyRunnerCfg):
    # ... 其他配置 ...
    
    # 启用扭矩记录
    enable_torque_recording: bool = True
    
    # 可选：自定义保存目录（默认为 "torque_logs"）
    torque_recording_dir: str = "torque_logs"
    
    # 可选：指定要记录的环境ID（默认为0，即第一个环境）
    torque_recording_env_id: int = 0
```

然后正常运行训练或推理：
```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task YourTask-v0
```

**注意**：命令行参数会覆盖配置文件中的设置

### 3. 录制操作

1. 程序启动后，您会看到提示信息：
   ```
   [TorqueRecorder] 扭矩记录器已启动
   [TorqueRecorder] 按 ',' 键开始录制，按 '.' 键结束录制并保存
   ```

2. 在需要录制的时刻按下 `,` 键开始录制

3. 按下 `.` 键停止录制，数据会自动保存

### 4. 查看结果

录制完成后，会在指定目录下生成两个文件：

- `torque_env{env_id}_{timestamp}.npz`：原始数据文件
- `torque_env{env_id}_{timestamp}.png`：可视化图形

#### 数据文件内容

使用numpy加载数据：

```python
import numpy as np

data = np.load('torque_logs/torque_env0_20231122_143025.npz')
timestamps = data['timestamps']  # 时间戳数组
torques = data['torques']        # 扭矩数据，shape: (num_steps, num_joints)
joint_names = data['joint_names'] # 关节名称列表
```

## 示例配置

### G1机器人配置示例

```python
# 在 source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agent/rsl_rl_ppo_cfg.py

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass

@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "g1_flat"
    
    # 启用扭矩记录
    enable_torque_recording = True
    torque_recording_dir = "logs/torque_recordings/g1"
    torque_recording_env_id = 0
    
    # ... 其他配置 ...
```

## 注意事项

1. **性能影响**：录制过程会占用少量CPU资源，但不会显著影响训练/推理性能
2. **存储空间**：长时间录制会产生较大的数据文件，建议录制关键片段
3. **环境要求**：需要安装 `pynput` 和 `matplotlib` 库
4. **多环境**：默认只记录第一个环境（env_id=0）的数据，可通过 `torque_recording_env_id` 修改

## 依赖安装

如果缺少依赖，请安装：

```bash
pip install pynput matplotlib
```

## 故障排除

### 问题：提示找不到robot

**原因**：环境中没有名为 "robot" 的场景实体

**解决**：确保您的环境配置中有 `scene.robot` 对象

### 问题：按键无响应

**原因**：键盘监听器可能没有正确启动

**解决**：检查终端是否有权限访问键盘输入，或尝试在图形界面下运行

### 问题：图形无法保存

**原因**：matplotlib后端配置问题

**解决**：设置环境变量 `export MPLBACKEND=Agg` 或在代码中使用非交互式后端
