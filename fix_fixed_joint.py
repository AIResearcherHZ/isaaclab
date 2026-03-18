#!/usr/bin/env python3
"""删除USD文件中有问题的rootJoint_base_link以消除PhysX警告"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="修复USD文件中的FixedJoint警告")
parser.add_argument("--headless", action="store_true", help="无头模式运行")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics, Sdf

usd_path = "/home/xhz/isaaclab/source/isaaclab_assets/isaaclab_assets/robots/double_pendulum/double_pendulum.usd"

print(f"正在修复USD文件: {usd_path}")
stage = Usd.Stage.Open(usd_path)

# 删除有问题的rootJoint_base_link
root_joint_path = "/double_pendulum/joints/rootJoint_base_link"
root_joint_prim = stage.GetPrimAtPath(root_joint_path)

if root_joint_prim:
    print(f"\n找到有问题的关节: {root_joint_path}")
    joint = UsdPhysics.Joint(root_joint_prim)
    print(f"  body0: {joint.GetBody0Rel().GetTargets()}")
    print(f"  body1: {joint.GetBody1Rel().GetTargets()}")
    
    # 删除这个prim
    print(f"\n删除关节prim...")
    stage.RemovePrim(root_joint_path)
    print("  已删除")
else:
    print(f"未找到关节: {root_joint_path}")

# 验证删除结果
print("\n验证删除后的关节列表:")
joints_scope = stage.GetPrimAtPath("/double_pendulum/joints")
if joints_scope:
    for child in joints_scope.GetChildren():
        print(f"  {child.GetPath()} ({child.GetTypeName()})")

print("\n保存修复后的文件...")
stage.Save()
print("完成! rootJoint_base_link已被删除，PhysX警告应该消失。")

simulation_app.close()
