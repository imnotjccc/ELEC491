# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import cv2
import hydra
import pybullet as p
import pybulletX as px
import tacto  # Import TACTO

log = logging.getLogger(__name__)


# Load the config YAML file from examples/conf/digit.yaml
@hydra.main(config_path="conf", config_name="digit")
def main(cfg):
    # Initialize digits
    bg = cv2.imread("conf/bg_digit_240_320.jpg")
    #digits = tacto.Sensor(**cfg.tacto, background=bg)

    #     :param width: scalar
    #     :param height: scalar
    #     :param background: image
    #     :param visualize_gui: Bool
    #     :param show_depth: Bool
    #     :param config_path:
    #     :param cid: Int
    digits = tacto.Sensor(
        width=120,
        height=160,
        background=bg,
        config_path="../meshes/case_meshes/config_sensor_case.yml",
        visualize_gui=True,
        show_depth=True,
        zrange=0.002,
        cid=0,
    )

    # Initialize World
    log.info("Initializing world")
    px.init()

    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    # Create and initialize DIGIT
    #digit_body = px.Body(**cfg.digit)
    digit_body = px.Body(
        urdf_path="../meshes/case_meshes/case_with_sensor.urdf",
        base_position=[0, 0, 0.075],
        base_orientation= [0.70710678, 0, 0, -0.70710678] ,
        use_fixed_base=True,
    )
    print("-----URDF INFO-----")
    print("base link name:", p.getBodyInfo(digit_body.id)[0].decode("utf-8"))  # base 的名字
    n = p.getNumJoints(digit_body.id)
    for i in range(n):
        ji = p.getJointInfo(digit_body.id, i)
        joint_name = ji[1].decode("utf-8")
        link_name  = ji[12].decode("utf-8")   # 这个字段是 child link name
        parent     = ji[16]                   # parent link index
        joint_type = ji[2]
        print(f"linkIndex={i:2d} link={link_name:20s}  joint={joint_name:20s}  parent={parent} type={joint_type}")
    digits.add_camera(digit_body.id, [0])

    # Add object to pybullet and tacto simulator
    #obj = px.Body(**cfg.object)
    obj = px.Body(
        urdf_path="objects/sphere_small.urdf",
        base_position=[-0.015, 0, 0.15],
        base_orientation=[0.0, 0.0, 0.0, 1.0],
        use_fixed_base=False,
        global_scaling=0.3
    )
    digits.add_body(obj)

    # Create control panel to control the 6DoF pose of the object
    panel = px.gui.PoseControlPanel(obj, **cfg.object_control_panel)
    panel.start()
    log.info("Use the slides to move the object until in contact with the DIGIT")

    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    while True:
        color, depth = digits.render()
        digits.updateGUI(color, depth)

    t.stop()


if __name__ == "__main__":
    main()
