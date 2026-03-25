# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import cv2
import hydra
import pybullet as p
import pybulletX as px
import tacto  # Import TACTO
from forceDisplacementPlotter import ForceDisplacementPlotter

log = logging.getLogger(__name__)


# Load the config YAML file from examples/conf/digit.yaml
@hydra.main(config_path="conf", config_name="digit")
def main(cfg):
    # Initialize digits
    bg = cv2.imread("conf/bg_digit_240_320.jpg")
    digits = tacto.Sensor(**cfg.tacto, background=bg)

    # Initialize World
    log.info("Initializing world")
    px.init()

    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    # Create and initialize DIGIT
    digit_body = px.Body(**cfg.digit)
    
    # Setting Penetration Depth
    # F=k⋅d+c⋅v
    stiffness = 2000.0  # Stiffness (N/m)：Determing the force it need to apply on
                        #                  the object to penetrate into the rigid body.
                        #                  Smaller value, easier to penetrate.
    damping = 100.0     # Damping：Prevent high frequency oscillation of the object 
                        #           or flip away the object when penetrating
    p.changeDynamics(
        bodyUniqueId=digit_body.id, 
        linkIndex=-1, 
        contactStiffness=stiffness, 
        contactDamping=damping
    )

    digits.add_camera(digit_body.id, [-1])

    # # Add object to pybullet and tacto simulator
    # obj = px.Body(**cfg.object)
    # digits.add_body(obj)

    # # Create control panel to control the 6DoF pose of the object
    # panel = px.gui.PoseControlPanel(obj, **cfg.object_control_panel)
    # panel.start()
    # log.info("Use the slides to move the object until in contact with the DIGIT")

    step = 1e-4
    z_min = 0.02
    z_max = z_min + 0.006
    z0 = z_max
        
    direction = -1

    obj = px.Body(
        urdf_path="objects/sphere_small.urdf",
        base_position=[-0.015, 0.0, z0],
        base_orientation=[0.0, 0.0, 0.0, 1.0],
        use_fixed_base=False,
        global_scaling=0.3
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

    digits.add_body(obj)

    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    plotter = ForceDisplacementPlotter(k_theoretical=1283.5)

    while True:
        z0 += direction * step
        if z0 >= z_max:
            z0 = z_max; direction = -1
        elif z0 <= z_min:
            z0 = z_min; #direction = +1
        p.changeConstraint(cid, jointChildPivot=[x0, y0, z0], maxForce=15)
        p.stepSimulation()

        color, depth = digits.render()
        digits.updateGUI(color, depth)

        F, delta_max, delta_min = digits.renderer.estimate_force_from_tacto_depth(depth)
        if F != 0:
            plotter.update_plot(z_current=z0, f_current=F)

    t.stop()


if __name__ == "__main__":
    main()
