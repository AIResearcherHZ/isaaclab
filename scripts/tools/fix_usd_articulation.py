#!/usr/bin/env python3
"""修复USD文件缺少ArticulationRootAPI的问题

当URDF转换为USD后，如果机器人无法被Isaac Sim识别为Articulation，
通常是因为根link缺少ArticulationRootAPI。此脚本自动检测并修复该问题。

使用方法:
    ./isaaclab.sh -p scripts/tools/fix_usd_articulation.py <usd_path> [--root_prim <prim_path>]

示例:
    # 自动检测根link并修复
    ./isaaclab.sh -p scripts/tools/fix_usd_articulation.py \
        /path/to/robot.usd

    # 指定根link路径
    ./isaaclab.sh -p scripts/tools/fix_usd_articulation.py \
        /path/to/robot.usd --root_prim /Robot/pelvis
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="修复USD文件缺少ArticulationRootAPI的问题")
    parser.add_argument("usd_path", type=str, help="USD文件路径")
    parser.add_argument("--root_prim", type=str, default=None, help="指定根link的prim路径（可选，默认自动检测）")
    args = parser.parse_args()

    # 启动Isaac Sim
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    from pxr import Usd, UsdPhysics

    # 打开USD文件
    stage = Usd.Stage.Open(args.usd_path)
    if not stage:
        print(f"[ERROR] 无法打开USD文件: {args.usd_path}")
        app.close()
        sys.exit(1)

    default_prim = stage.GetDefaultPrim()
    print(f"[INFO] 默认prim: {default_prim.GetPath()}")

    # 确定要添加ArticulationRootAPI的prim
    target_prim = None

    if args.root_prim:
        # 使用用户指定的路径
        target_prim = stage.GetPrimAtPath(args.root_prim)
        if not target_prim.IsValid():
            print(f"[ERROR] 指定的prim路径无效: {args.root_prim}")
            app.close()
            sys.exit(1)
    else:
        # 自动检测：查找第一个有RigidBodyAPI的prim
        def find_rigid_body_prim(prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return prim
            for child in prim.GetChildren():
                result = find_rigid_body_prim(child)
                if result:
                    return result
            return None

        target_prim = find_rigid_body_prim(default_prim)

        if not target_prim:
            print("[ERROR] 未找到带有RigidBodyAPI的prim，请使用--root_prim手动指定")
            app.close()
            sys.exit(1)

    print(f"[INFO] 目标prim: {target_prim.GetPath()}")

    # 检查是否已有ArticulationRootAPI
    if target_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print(f"[INFO] {target_prim.GetPath()} 已有ArticulationRootAPI，无需修复")
    else:
        # 添加ArticulationRootAPI
        UsdPhysics.ArticulationRootAPI.Apply(target_prim)
        print(f"[SUCCESS] 已为 {target_prim.GetPath()} 添加ArticulationRootAPI")

    # 保存修改
    stage.Save()
    print(f"[SUCCESS] USD文件已保存: {args.usd_path}")

    # 验证
    target_prim = stage.GetPrimAtPath(target_prim.GetPath())
    print(f"[VERIFY] ArticulationRootAPI存在: {target_prim.HasAPI(UsdPhysics.ArticulationRootAPI)}")

    app.close()


if __name__ == "__main__":
    main()
