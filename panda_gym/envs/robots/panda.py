import numpy as np
from gym import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet
import tacto
import pybullet as p

class Panda(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optionnal): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
        control_type: str = "ee_orn",
    ) -> None:
        self.block_gripper = block_gripper
        self.control_type = control_type
        # n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        if self.control_type == "ee":
            n_action = 3
        elif self.control_type == "ee_orn":
            n_action = 6
        else:
            n_action = 7
        # print(self.control_type)
        # if self.control_type == "ee_orn":
        #     n_action = 6
        # else:
        #     n_action = 7

        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="panda",
            file_name="panda_gym/envs/robots/franka_panda/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            arm_indices=np.array([4, 5, 6]), # define arm indices besides finger joints
            ee_index = np.array([11]), # define end-effector link index
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )
        # debug urdf
        # print("-----URDF INFO-----")
        # print("base link name:", p.getBodyInfo(self.sim._bodies_idx["panda"])[0].decode("utf-8"))  # base 的名字
        # n = p.getNumJoints(self.sim._bodies_idx["panda"])
        # for i in range(n):
        #     ji = p.getJointInfo(self.sim._bodies_idx["panda"], i)
        #     joint_name = ji[1].decode("utf-8")
        #     link_name  = ji[12].decode("utf-8")   # 这个字段是 child link name
        #     parent     = ji[16]                   # parent link index
        #     joint_type = ji[2]
        #     print(f"linkIndex={i:2d} link={link_name:20s}  joint={joint_name:20s}  parent={parent} type={joint_type}")

        self.fingers_indices = np.array([9, 10])

        self.ee_link = 11

        # self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.neutral_joint_values = np.array([-0.01379803, -0.4221698, 0.0081434, -2.62930775, 0.00414516, 2.2071252, 0.77655979])

        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

        # init tacto sensor
        self.tactileSensor_ee = tacto.Sensor(
            width=120,
            height=160,
            background=None,
            config_path="tacto/meshes/case_meshes/config_sensor_case.yml",
            visualize_gui=True,
            show_depth=True,
            zrange=0.0002, #0.002
            cid=0,
        )
        self.sim.tatco_set_penetration_depth(
            body = "panda",
            link = 12,
            stiffness = 1900.0,
            damping = 100.0
        )
        self.tactileSensor_ee.add_camera(self.sim._bodies_idx["panda"], [12])

        # debug gel shape
        vis = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=self.tactileSensor_ee.renderer.conf.sensor.gel.mesh,   # 或 .stl
            meshScale=[1, 1, 1],
            rgbaColor=[1, 1, 0, 0.4],               # 半透明黄
        )
        self.gel_vis_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis,
            baseCollisionShapeIndex=-1,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
        )

        #p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
            # current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
            # print(current_arm_joint_angles)
        elif self.control_type == "ee_orn":
            ee_displacement = action[:6]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        self.control_joints(target_angles=target_angles)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        if self.control_type == "ee":
            ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
            # get the current position and the target position
            ee_position = self.get_ee_position()
            target_ee_position = ee_position + ee_displacement
            # print("target_ee_position:", target_ee_position)
            # Clip the height target. For some reason, it has a great impact on learning
            target_ee_position[2] = np.max((0, target_ee_position[2]))
            # compute the new joint angles
            target_arm_angles = self.inverse_kinematics(
                link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
            )
            # print(target_arm_angles)
            target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        elif self.control_type == "ee_orn":
            ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
            ee_orn_ctrl = ee_displacement[3:] * 0.05  # limit maximum change in orientation
            # get the current position and the target position
            ee_position = self.get_ee_position()
            target_ee_position = ee_position + ee_displacement[:3]
            # Clip the height target. For some reason, it has a great impact on learning
            target_ee_position[2] = np.max((0, target_ee_position[2]))
            # compute the new joint angles
            current_ee_orn = p.getLinkState(self.sim._bodies_idx[self.body_name], self.ee_link)[1]
            target_ee_orn = p.getQuaternionFromEuler(np.array(p.getEulerFromQuaternion(current_ee_orn)) + ee_orn_ctrl)
            target_arm_angles = self.inverse_kinematics(
                link=self.ee_link, position=target_ee_position, orientation=target_ee_orn
            )
            target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles
        

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        contact_force = self.get_contact_force()

        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            obs = np.concatenate((ee_position, ee_velocity, [fingers_width], contact_force))
        else:
            obs = np.concatenate((ee_position, ee_velocity, contact_force))
        return obs

    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the ned-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)
    
    # add by me
    def get_joint_velocity(self, joint: int) -> float:
        """Retuens the velocity of arm joints."""
        arm_link_vel = []
        for i in self.arm_indices:
            arm_link_vel.append(self.sim.get_joint_velocity(self.body_name, i))
        return np.array(arm_link_vel)
    
    def getPosAndOrnFromMatrix(self, matrix):
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
    
    def get_contact_force(self):
        # get contact force
        color, depth = self.tactileSensor_ee.render() # update tacto sensor
   
        F, delta_max, delta_mean = self.tactileSensor_ee.renderer.estimate_force_from_tacto_depth(depth, 
                                                                                                d_max = 5e-3, 
                                                                                                d_min=1e-6, 
                                                                                                f_offset = 0.0, 
                                                                                                k = 2000.0
                                                                                                )
        # self.tactileSensor_ee.updateGUI(color, depth)
        F = np.array(list(F.values()), dtype=np.float32)
        # print(F)
        # print(f"F_cam0 = {F[0]} Max_cam0 = {delta_max[0]} Mean_cam0 = {delta_mean[0]}")
        # print(f"F_cam1 = {F[1]} Max_cam1 = {delta_max[1]} Mean_cam1 = {delta_mean[1]}")
        return F