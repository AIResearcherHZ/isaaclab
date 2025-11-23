# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""扭矩记录器模块，用于记录和可视化关节扭矩数据"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from pynput import keyboard

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


class TorqueRecorder:
    """扭矩记录器类，支持键盘控制的扭矩数据记录和可视化

    使用方法：
    - 按 ',' 键开始录制
    - 按 '.' 键结束录制并保存数据
    """

    def __init__(
        self,
        articulation: Articulation,
        save_dir: str = "torque_logs",
        env_id: int = 0,
        enable: bool = True,
    ):
        """初始化扭矩记录器

        Args:
            articulation: 机器人关节对象
            save_dir: 保存目录
            env_id: 要记录的环境ID（默认为0，即第一个环境）
            enable: 是否启用记录功能
        """
        self.articulation = articulation
        self.save_dir = save_dir
        self.env_id = env_id
        self.enable = enable

        # 获取关节名称
        self.joint_names = articulation.data.joint_names
        self.num_joints = len(self.joint_names)

        # 记录状态
        self.is_recording = False
        self.recorded_data = []
        self.timestamps = []
        self.start_time = None

        # 创建保存目录
        if self.enable:
            os.makedirs(self.save_dir, exist_ok=True)

            # 启动键盘监听
            self.listener = keyboard.Listener(on_press=self._on_key_press)
            self.listener.start()

            print("[TorqueRecorder] 扭矩记录器已启动")
            print("[TorqueRecorder] 按 ',' 键开始录制，按 '.' 键结束录制并保存")

    def _on_key_press(self, key):
        """键盘按键回调"""
        try:
            if hasattr(key, 'char'):
                if key.char == ',':
                    self._start_recording()
                elif key.char == '.':
                    self._stop_recording()
        except AttributeError:
            pass

    def _start_recording(self):
        """开始录制"""
        if not self.is_recording:
            self.is_recording = True
            self.recorded_data = []
            self.timestamps = []
            self.start_time = None
            print(f"\n[TorqueRecorder] 开始录制扭矩数据 (环境 {self.env_id})...")

    def _stop_recording(self):
        """停止录制并保存数据"""
        if self.is_recording:
            self.is_recording = False
            print(f"[TorqueRecorder] 停止录制，共记录 {len(self.recorded_data)} 个数据点")

            if len(self.recorded_data) > 0:
                # 在新线程中保存，避免阻塞主循环
                threading.Thread(target=self._save_and_plot, daemon=True).start()
            else:
                print("[TorqueRecorder] 没有数据可保存")

    def record_step(self):
        """记录当前时间步的扭矩数据"""
        if not self.enable or not self.is_recording:
            return

        # 获取当前扭矩数据（只记录指定环境的数据）
        applied_torque = self.articulation.data.applied_torque[self.env_id].cpu().numpy()

        # 记录时间戳
        if self.start_time is None:
            self.start_time = 0.0
            timestamp = 0.0
        else:
            timestamp = len(self.recorded_data) * self.articulation._sim_dt

        self.timestamps.append(timestamp)
        self.recorded_data.append(applied_torque.copy())

    def _save_and_plot(self):
        """保存数据并绘制曲线"""
        if len(self.recorded_data) == 0:
            return

        # 转换为numpy数组
        data_array = np.array(self.recorded_data)  # shape: (num_steps, num_joints)
        timestamps = np.array(self.timestamps)

        # 生成时间戳文件名
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"torque_env{self.env_id}_{timestamp_str}"

        # 保存原始数据
        data_file = os.path.join(self.save_dir, f"{base_filename}.npz")
        np.savez(
            data_file,
            timestamps=timestamps,
            torques=data_array,
            joint_names=self.joint_names,
        )
        print(f"[TorqueRecorder] 数据已保存至: {data_file}")

        # 绘制曲线
        self._plot_torques(timestamps, data_array, base_filename)

    def _plot_torques(self, timestamps, data_array, base_filename):
        """绘制扭矩曲线"""
        num_joints = data_array.shape[1]

        # 计算子图布局
        n_cols = min(3, num_joints)
        n_rows = (num_joints + n_cols - 1) // n_cols

        # 创建图形
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if num_joints == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # 绘制每个关节的扭矩曲线
        for i in range(num_joints):
            ax = axes[i]
            ax.plot(timestamps, data_array[:, i], linewidth=1.5)
            ax.set_xlabel('时间 (s)', fontsize=10)
            ax.set_ylabel('扭矩 (N·m)', fontsize=10)
            ax.set_title(f'{self.joint_names[i]}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # 添加统计信息
            mean_val = np.mean(data_array[:, i])
            max_val = np.max(data_array[:, i])
            min_val = np.min(data_array[:, i])
            ax.text(
                0.02, 0.98,
                f'均值: {mean_val:.2f}\n最大: {max_val:.2f}\n最小: {min_val:.2f}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8
            )

        # 隐藏多余的子图
        for i in range(num_joints, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # 保存图形
        plot_file = os.path.join(self.save_dir, f"{base_filename}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"[TorqueRecorder] 图形已保存至: {plot_file}")

        plt.close(fig)

    def close(self):
        """关闭记录器"""
        if self.enable:
            if self.is_recording:
                self._stop_recording()
            if hasattr(self, 'listener'):
                self.listener.stop()
            print("[TorqueRecorder] 扭矩记录器已关闭")
