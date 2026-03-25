# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import cv2
import numpy as np
import hydra
import pybullet as p
import pybulletX as px
import tacto  # Import TACTO
from forceDisplacementPlotter import ForceDisplacementPlotter
import time

log = logging.getLogger(__name__)

def getPosAndOrnFromMatrix(matrix):
    """Extract position and orientation quaternion from a transform matrix.

    Accepts a 4x4 homogeneous matrix (or a flat list/array of length 16) or a 3x4
    transform (rows are [R|t]). Returns (position, quaternion) where quaternion is
    in pybullet format [x, y, z, w].
    """
    m = np.array(matrix, dtype=float)
    # reshape flat 16-length to 4x4
    if m.size == 16:
        m = m.reshape(4, 4)
    # accept 3x4 (R|t) or 4x4 homogeneous
    if m.shape == (4, 4):
        R = m[:3, :3]
        pos = m[:3, 3]
    elif m.shape == (3, 4):
        R = m[:, :3]
        pos = m[:, 3]
    else:
        raise ValueError(f"Unsupported matrix shape {m.shape}, expected (4,4) or (3,4)")

    r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]

    trace = r00 + r11 + r22
    if trace > 0.0:
        S = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * S
        qx = (r21 - r12) / S
        qy = (r02 - r20) / S
        qz = (r10 - r01) / S
    else:
        if (r00 > r11) and (r00 > r22):
            S = np.sqrt(1.0 + r00 - r11 - r22) * 2.0
            qw = (r21 - r12) / S
            qx = 0.25 * S
            qy = (r01 + r10) / S
            qz = (r02 + r20) / S
        elif r11 > r22:
            S = np.sqrt(1.0 + r11 - r00 - r22) * 2.0
            qw = (r02 - r20) / S
            qx = (r01 + r10) / S
            qy = 0.25 * S
            qz = (r12 + r21) / S
        else:
            S = np.sqrt(1.0 + r22 - r00 - r11) * 2.0
            qw = (r10 - r01) / S
            qx = (r02 + r20) / S
            qy = (r12 + r21) / S
            qz = 0.25 * S

    # pybullet uses [x, y, z, w]
    quat = [float(qx), float(qy), float(qz), float(qw)]
    pos = [float(p) for p in pos]
    return pos, quat

def draw_camera_coordinate_system(pos, quat, length=0.05, lifeTime=0.1):
    """
    在 PyBullet 中绘制相机的局部坐标系
    :param pos: 相机的世界坐标 [x, y, z]
    :param quat: 相机的四元数姿态 [x, y, z, w]
    :param length: 坐标轴显示的长度 (单位: m)
    :param lifeTime: 线条持续时间(秒)。设为 0 表示永久保留。如果在循环中调用，建议设为 0.1 等短时间避免画面卡顿
    """
    # 1. 将四元数转换为 3x3 的旋转矩阵
    rot_matrix = p.getMatrixFromQuaternion(quat)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)

    # 2. 提取局部坐标系的三个轴在世界坐标系下的方向向量 (旋转矩阵的列)
    x_axis = rot_matrix[:, 0]
    y_axis = rot_matrix[:, 1]
    z_axis = rot_matrix[:, 2]

    # 3. 计算三个轴的终点坐标
    pos = np.array(pos)
    end_x = pos + x_axis * length
    end_y = pos + y_axis * length
    end_z = pos + z_axis * length

    # 4. 绘制 RGB 线条对应 XYZ
    # 红色 - X轴
    p.addUserDebugLine(pos.tolist(), end_x.tolist(), lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=lifeTime)
    # 绿色 - Y轴
    p.addUserDebugLine(pos.tolist(), end_y.tolist(), lineColorRGB=[0, 1, 0], lineWidth=2, lifeTime=lifeTime)
    # 蓝色 - Z轴
    p.addUserDebugLine(pos.tolist(), end_z.tolist(), lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=lifeTime)
    
    # 5. 在末端添加文字标签，方便直接看出来
    p.addUserDebugText("X", end_x.tolist(), textColorRGB=[1, 0, 0], textSize=1.2, lifeTime=lifeTime)
    p.addUserDebugText("Y", end_y.tolist(), textColorRGB=[0, 1, 0], textSize=1.2, lifeTime=lifeTime)
    p.addUserDebugText("Z", end_z.tolist(), textColorRGB=[0, 0, 1], textSize=1.2, lifeTime=lifeTime)

# Load the config YAML file from examples/conf/digit.yaml
@hydra.main(config_path="conf", config_name="digit")
def main(cfg):
    Debug_force = False

    # Initialize digits
    #     :param width: scalar
    #     :param height: scalar
    #     :param background: image
    #     :param visualize_gui: Bool
    #     :param show_depth: Bool
    #     :param config_path:
    #     :param cid: Int
    tactileSensor_ee = tacto.Sensor(
        width=120,
        height=160,
        background=None,
        config_path="../meshes/case_meshes/config_sensor_case.yml",
        visualize_gui=True,
        show_depth=True,
        zrange=0.0002, #0.002
        cid=0,
    )

    # Initialize World
    log.info("Initializing world")
    px.init()

    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    # Create and initialize DIGIT
    franka_body = px.Body(
        urdf_path="../../panda_gym/envs/robots/franka_panda/panda.urdf",
        base_position=[0, 0, 0.0], # [0, 0, 0.075]
        base_orientation= [0, 0, 0, 1] ,
        use_fixed_base=True,
    )

    print("-----URDF INFO-----")
    print("base link name:", p.getBodyInfo(franka_body.id)[0].decode("utf-8"))  # base 的名字
    n = p.getNumJoints(franka_body.id)
    for i in range(n):
        ji = p.getJointInfo(franka_body.id, i)
        joint_name = ji[1].decode("utf-8")
        link_name  = ji[12].decode("utf-8")   # 这个字段是 child link name
        parent     = ji[16]                   # parent link index
        joint_type = ji[2]
        print(f"linkIndex={i:2d} link={link_name:20s}  joint={joint_name:20s}  parent={parent} type={joint_type}")
    tactileSensor_ee.add_camera(franka_body.id, [12])

    # debug gel shape
    vis = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=tactileSensor_ee.renderer.conf.sensor.gel.mesh,   # 或 .stl
        meshScale=[1, 1, 1],
        rgbaColor=[1, 1, 0, 0.4],               # 半透明黄
    )
    gel_vis_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=vis,
        baseCollisionShapeIndex=-1,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
    )

    if Debug_force:
        step = 1e-5
        y_min = -0.0766
        y_max = y_min + 0.005
        y0 = y_min
        
        direction = +1
        obj = px.Body(
            urdf_path="objects/sphere_small.urdf",
            base_position=[0.027, y0, 0.98],
            base_orientation=[0.0, 0.0, 0.0, 1.0],
            use_fixed_base=False,
            global_scaling=0.5
        )

        print("base pos/orn =", p.getBasePositionAndOrientation(obj.id))
        obj_pos0, obj_orn0 = p.getBasePositionAndOrientation(obj.id)
        x0, y0, z0 = obj_pos0

        cid = p.createConstraint(
            parentBodyUniqueId=obj.id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,        # world
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],     # 在球的局部坐标：球心
            childFramePosition=[x0, y0, z0],   # 在世界坐标：目标点
            parentFrameOrientation=obj_orn0,   # 可选：想锁定姿态就给
            childFrameOrientation=[0, 0, 0, 1]
        )
        p.changeConstraint(cid, maxForce=20)

        tactileSensor_ee.add_body(obj)

    else:
        # Add object to pybullet and tacto simulator
        obj = px.Body(
            urdf_path="objects/sphere_small.urdf",
            base_position=[0.027, -0.074, 0.98],
            base_orientation=[0.0, 0.0, 0.0, 1.0],
            use_fixed_base=False,
            global_scaling=0.3
        )
        tactileSensor_ee.add_body(obj)
        # Create control panel to control the 6DoF pose of the object
        panel = px.gui.PoseControlPanel(obj, **cfg.object_control_panel)
        panel.start()
        log.info("Use the slides to move the object until in contact with the DIGIT")

    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    plotter = ForceDisplacementPlotter(k_theoretical=1283.5)

    color, depth = tactileSensor_ee.render()
    tactileSensor_ee.updateGUI(color, depth)

    while True:
        p.stepSimulation()

        gel_matrix = tactileSensor_ee.renderer.gel_node.matrix
        gel_pos, gel_orn = getPosAndOrnFromMatrix(gel_matrix)
        p.resetBasePositionAndOrientation(gel_vis_id, gel_pos, gel_orn)  # [web:452]

        for nb_cam in range(tactileSensor_ee.renderer.nb_cam):
            cam_matrix = tactileSensor_ee.renderer.camera_nodes[nb_cam].matrix
            cam_pos, gel_orn = getPosAndOrnFromMatrix(cam_matrix)
            draw_camera_coordinate_system(cam_pos, gel_orn, length=0.1)
        
        if  Debug_force:
            # 每帧更新目标z
            y0 += direction * step
            if y0 >= y_max:
                y0 = y_max; direction = -1
            elif y0 <= y_min:
                y0 = y_min; direction = +1
            p.changeConstraint(cid, jointChildPivot=[x0, y0, z0], maxForce=20)

            F, delta_max, delta_min = tactileSensor_ee.renderer.estimate_force_from_tacto_depth(depth)
            if F != 0:
                plotter.update_plot(z_current=abs(y0), f_current=F)
                print(f"y = {y0} f = {F}")

        color, depth = tactileSensor_ee.render()
        tactileSensor_ee.updateGUI(color, depth)
    t.stop()
    plotter.keep_window_open()

if __name__ == "__main__":
    main()
