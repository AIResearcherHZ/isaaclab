# 扭矩记录功能使用说明

## 功能概述

扭矩记录功能允许您在训练和推理过程中实时记录机器人关节的扭矩数据，并自动生成可视化曲线图。

## 主要特性

- **自动录制**：程序启动后自动开始录制，无需手动操作
- **固定时长**：录制指定时长（默认5秒）后自动保存
- **Ctrl+C保存**：退出时自动保存已录制的数据
- **Rich终端界面**：使用rich库提供美观的终端显示，包括启动面板、录制状态、数据统计表格等，特别适合服务器环境
- **实时状态反馈**：在终端清晰显示录制状态、数据点数、时长等信息
- **自适应关节名称**：自动获取环境中的关节名称，无需手动配置
- **可视化**：自动生成扭矩曲线图，包含统计信息（均值、最大值、最小值）
- **数据保存**：保存原始数据（.npz格式）和图形（.png格式）
- **详细统计**：保存完成后在终端显示每个关节的均值、最大值、最小值和标准差

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

#### 自定义保存目录、环境ID和录制时长
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-G1-Play-v0 \
    --num_envs 1 \
    --enable_torque_recording \
    --torque_recording_dir logs/my_torque_data \
    --torque_recording_env_id 0 \
    --torque_recording_duration 5.0
```

**参数说明**：
- `--torque_recording_duration`: 录制时长（秒），默认5.0秒

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

1. 程序启动后，会自动开始录制，终端显示启动面板（使用rich库渲染）：
   ```
   ╭─────────────── 🎬 扭矩记录器已启动 ───────────────╮
   │                                                   │
   │  ⚙️  配置信息                                      │
   │  环境ID       0                                   │
   │  关节数       23                                  │
   │  录制时长     5.0秒                               │
   │  保存目录     logs/my_torque_data                 │
   │                                                   │
   │  🔴 自动录制中... (Ctrl+C 退出时自动保存)          │
   │                                                   │
   ╰───────────────────────────────────────────────────╯
   ```

2. 录制达到指定时长后自动保存，终端会显示：
   ```
   � 正在保存数据...
   数据点数: 250 | 时长: 5.00秒
   ✅ 数据已保存: logs/my_torque_data/torque_env0_20231123_183025.npz
   ✅ 图形已保存: logs/my_torque_data/torque_env0_20231123_183025.png
   ```

3. 如果在录制期间按 Ctrl+C 退出，会自动保存已录制的数据：
   ```
   ⚠️  检测到退出信号，正在保存数据...
   💾 正在保存数据...
   数据点数: 150 | 时长: 3.00秒
   ✅ 数据已保存: logs/my_torque_data/torque_env0_20231123_183025.npz
   ✅ 图形已保存: logs/my_torque_data/torque_env0_20231123_183025.png
   ```

4. 保存完成后，会显示一个详细的数据统计表格：
   ```
   ╭─────────────── 📊 扭矩数据统计 ───────────────╮
   │ 关节名称        均值    最大值   最小值   标准差 │
   │ left_hip_yaw    2.45    8.32    -3.21    1.87  │
   │ left_hip_roll   1.23    5.67    -2.34    1.12  │
   │ ...                                            │
   ╰────────────────────────────────────────────────╯
   ```

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

1. **自动录制**：程序启动后立即开始录制，无需手动操作
2. **录制时长**：默认录制5秒，可通过 `--torque_recording_duration` 参数自定义
3. **性能影响**：录制过程会占用少量CPU资源，但不会显著影响训练/推理性能
4. **存储空间**：长时间录制会产生较大的数据文件，建议设置合理的录制时长
5. **环境要求**：需要安装 `matplotlib` 和 `rich` 库
6. **多环境**：默认只记录第一个环境（env_id=0）的数据，可通过 `torque_recording_env_id` 修改
7. **终端显示**：使用 `rich` 库提供美观的终端界面，特别适合服务器环境使用
8. **Ctrl+C保存**：任何时候按 Ctrl+C 退出都会自动保存已录制的数据

## 依赖安装

如果缺少依赖，请安装：

```bash
pip install matplotlib rich
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
