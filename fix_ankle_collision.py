#!/usr/bin/env python3
"""将USD文件中 *_ankle_roll_link 的碰撞近似设置为 convex decomposition"""

import argparse
import re
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="修复USD文件中ankle_roll_link的碰撞近似")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--usd", type=str, default="/home/xhz/桌面/assets/Taks_T1/Taks_T1.usd", help="USD文件路径")
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

usd_path = args_cli.usd
print(f"正在修复USD文件: {usd_path}")
stage = Usd.Stage.Open(usd_path)

ankle_pattern = re.compile(r".*_ankle_roll_link|pelvis|neck_pitch_link")
modified_count = 0

# 先打印所有ankle_roll_link相关的prim结构
print("\n=== 分析USD结构 ===")
for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    if ankle_pattern.search(path_str):
        print(f"  {path_str} -> 类型: {prim.GetTypeName()}")

# 遍历所有prim，找到ankle_roll_link/collisions下的Xform并设置碰撞近似
print("\n=== 设置碰撞近似 ===")
for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    
    # 检查是否是ankle_roll_link下的collisions Xform
    if ankle_pattern.search(path_str) and path_str.endswith("/collisions"):
        print(f"\n找到碰撞Xform: {path_str}")
        
        # 对collisions Xform本身应用MeshCollisionAPI
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")
        print(f"  设置 approximation = convexDecomposition")
        modified_count += 1
        
        # 同时遍历其所有子prim
        for child in prim.GetAllChildren():
            child_path = str(child.GetPath())
            child_type = child.GetTypeName()
            print(f"  子节点: {child_path} -> {child_type}")
            
            # 如果子节点有CollisionAPI，也设置其碰撞近似
            if child.HasAPI(UsdPhysics.CollisionAPI):
                child_mesh_api = UsdPhysics.MeshCollisionAPI.Apply(child)
                child_mesh_api.GetApproximationAttr().Set("convexDecomposition")
                print(f"    设置子节点 approximation = convexDecomposition")

print(f"\n共修改了 {modified_count} 个碰撞体")
print("\n保存修复后的文件...")
stage.Save()
print("完成!")

simulation_app.close()
