# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""扭矩记录器模块，用于记录和可视化关节扭矩数据"""

from __future__ import annotations

import os
import time
import warnings
from datetime import datetime
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# 配置 matplotlib 使用支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Bitstream Vera Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 抑制中文字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


class TorqueRecorder:
    """扭矩记录器类，自动录制指定时长的扭矩数据

    使用方法：
    - 程序启动后自动开始录制
    - 录制指定时长后自动保存
    - Ctrl+C 退出时也会自动保存已录制的数据
    """

    def __init__(
        self,
        articulation: Articulation,
        save_dir: str = "torque_logs",
        env_id: int = 0,
        enable: bool = True,
        recording_duration: float = 5.0,
        dt: float = 0.02,
    ):
        """初始化扭矩记录器

        Args:
            articulation: 机器人关节对象
            save_dir: 保存目录
            env_id: 要记录的环境ID（默认为0，即第一个环境）
            enable: 是否启用记录功能
            recording_duration: 录制时长（秒），默认5秒
            dt: 仿真时间步长（秒），默认0.02秒
        """
        self.articulation = articulation
        self.save_dir = save_dir
        self.env_id = env_id
        self.enable = enable
        self.recording_duration = recording_duration
        self.dt = dt

        # 获取关节名称
        self.joint_names = articulation.data.joint_names
        self.num_joints = len(self.joint_names)

        # 记录状态
        self.is_recording = True  # 自动开始录制
        self.recorded_data = []
        self.timestamps = []
        self.start_time = None
        self.recording_start_time = time.time()
        self.has_saved = False

        # Rich控制台
        self.console = Console()

        # 创建保存目录
        if self.enable:
            os.makedirs(self.save_dir, exist_ok=True)
            # 显示启动信息
            self._display_startup_info()

    def _save_data(self):
        """保存数据"""
        if self.has_saved or len(self.recorded_data) == 0:
            return

        self.has_saved = True
        data_points = len(self.recorded_data)
        duration = self.timestamps[-1] if self.timestamps else 0

        self.console.print("\n[bold yellow]💾 正在保存数据...[/bold yellow]")
        self.console.print(f"[cyan]数据点数: {data_points} | 时长: {duration:.2f}秒[/cyan]")

        self._save_and_plot()

    def record_step(self):
        """记录当前时间步的扭矩数据"""
        if not self.enable or not self.is_recording:
            return

        # 检查是否达到录制时长
        elapsed_time = time.time() - self.recording_start_time
        if elapsed_time >= self.recording_duration:
            if not self.has_saved:
                self.is_recording = False
                self._save_data()
            return

        # 获取当前扭矩数据（只记录指定环境的数据）
        applied_torque = self.articulation.data.applied_torque[self.env_id].cpu().numpy()

        # 记录时间戳
        if self.start_time is None:
            self.start_time = 0.0
            timestamp = 0.0
        else:
            timestamp = len(self.recorded_data) * self.dt

        self.timestamps.append(timestamp)
        self.recorded_data.append(applied_torque.copy())

    def _save_and_plot(self):
        """保存数据并绘制曲线"""
        if len(self.recorded_data) == 0:
            self.console.print("[red]⚠️  没有数据可保存[/red]")
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
        self.console.print(f"[bold green]✅ 数据已保存:[/bold green] [cyan]{data_file}[/cyan]")

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
        self.console.print(f"[bold green]✅ 图形已保存:[/bold green] [cyan]{plot_file}[/cyan]")

        # 显示数据统计
        self._display_statistics(data_array)

        plt.close(fig)

    def _display_startup_info(self):
        """显示启动信息"""
        # 创建配置信息
        config_table = Table(show_header=False, box=None, padding=(0, 2))
        config_table.add_column(style="bold magenta")
        config_table.add_column(style="white")

        config_table.add_row("环境ID:", str(self.env_id))
        config_table.add_row("关节数:", str(self.num_joints))
        config_table.add_row("录制时长:", f"{self.recording_duration}秒")
        config_table.add_row("保存目录:", self.save_dir)

        # 组合面板
        panel_content = Table.grid(padding=1)
        panel_content.add_row("[bold white]⚙️  配置信息[/bold white]")
        panel_content.add_row(config_table)
        panel_content.add_row("")
        panel_content.add_row("[bold yellow]🔴 自动录制中... (Ctrl+C 退出时自动保存)[/bold yellow]")

        panel = Panel(
            panel_content,
            title="[bold green]🎬 扭矩记录器已启动[/bold green]",
            border_style="green",
            padding=(1, 2)
        )

        self.console.print("\n")
        self.console.print(panel)
        self.console.print("\n")

    def _display_statistics(self, data_array):
        """显示数据统计信息"""
        stats_table = Table(title="📊 扭矩数据统计", show_header=True, header_style="bold magenta")
        stats_table.add_column("关节名称", style="cyan", no_wrap=True)
        stats_table.add_column("均值 (N·m)", justify="right", style="green")
        stats_table.add_column("最大值 (N·m)", justify="right", style="red")
        stats_table.add_column("最小值 (N·m)", justify="right", style="blue")
        stats_table.add_column("标准差 (N·m)", justify="right", style="yellow")

        for i, joint_name in enumerate(self.joint_names):
            mean_val = np.mean(data_array[:, i])
            max_val = np.max(data_array[:, i])
            min_val = np.min(data_array[:, i])
            std_val = np.std(data_array[:, i])

            stats_table.add_row(
                joint_name,
                f"{mean_val:.2f}",
                f"{max_val:.2f}",
                f"{min_val:.2f}",
                f"{std_val:.2f}"
            )

        self.console.print("\n")
        self.console.print(stats_table)
        self.console.print("\n")

    def close(self):
        """关闭记录器"""
        if self.enable:
            if not self.has_saved and len(self.recorded_data) > 0:
                self.console.print("\n[bold yellow]⚠️  检测到退出信号，正在保存数据...[/bold yellow]")
                self._save_data()
            self.console.print("[bold red]🛑 扭矩记录器已关闭[/bold red]")
