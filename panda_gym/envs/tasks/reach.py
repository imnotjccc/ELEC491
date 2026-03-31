from typing import Any, Dict, Union

import numpy as np
import random

import pybulletX as px
import pybullet as p

from panda_gym.envs.core import Task
from panda_gym.utils import distance

class Reach(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.03,# meters
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position

        self.goal_range_low = np.array([0.05, -goal_range / 2, 0.40])
        self.goal_range_high = np.array([0.13, goal_range / 2, 0.55])

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        #original goal position (green)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.4]),
        )

        # create a new structure for pybulletx
        self.obj = px.Body(
            # urdf_path="tacto/Obsticle_box/urdf/Obsticle_box.urdf",
            urdf_path="tacto/obsticle_cylinder/urdf/Obsticle_cylinder.urdf",
            base_position=[-0.19, 0.0, 0.57],
            base_orientation=[0.5, -0.5, -0.5, 0.5],
            use_fixed_base=True,
            # use_fixed_base=False,
            global_scaling= 1.0
        )

        self.cid = p.createConstraint(
            parentBodyUniqueId=self.obj.id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,        # world
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],     # 在球的局部坐标：球心
            childFramePosition=[0.0, 0.0, 0.0],   # 在世界坐标：目标点
            parentFrameOrientation=[0.0, 0.0, 0.0, 1.0],   # 可选：想锁定姿态就给
            childFrameOrientation=[0, 0, 0, 1]
        )
        p.changeConstraint(self.cid, maxForce=20)

        #new goal poaition (red)
        # self.sim.create_sphere(
        #     body_name="red_goal",
        #     radius=0.02,
        #     mass=0.0,
        #     ghost=True,
        #     position=np.zeros(3),
        #     rgba_color=np.array([0.9, 0.1, 0.1, 0.4]),
        # )

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no task-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        # print(self.goal)
        # self.goal = np.array([-0.20, 0.0, 0.30]) # make goal fixed for testing
        # self.red_goal = self._sample_red_goal()
        # self.red_goal = self.goal.copy()
        # self.red_goal[0] -= 0.30 # make sure red goal is not
        # self.screen_pose = self.goal.copy()
        # self.screen_pose[0] -= 0.1

        # self.sim.set_base_pose("red_goal", self.red_goal, np.array([0.0, 0.0, 0.0, 1.0]))
        # self.sim.set_base_pose("screen", self.screen_pose, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0])) # make target the new goal

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        if random.randint(0,1) == 1:
            goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        else:
            goal = self.np_random.uniform(np.array([-0.15, -0.15, 0.30]),
                                        np.array([-0.25, 0.15, 0.55]))
        return goal
    
    # def _sample_red_goal(self) -> np.ndarray:
    #     """Randomize second goal."""
    #     self.red_goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
    #     self.red_goal[0] -= 0.10 # make sure red goal is not collide with original goal
    #     return self.red_goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal[:3], desired_goal[:3])
        return np.array(d < self.distance_threshold, dtype=np.float64)
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal = np.asarray(achieved_goal)
        desired_goal = np.asarray(desired_goal)

        if achieved_goal.ndim == 1:
            achieved_pos = achieved_goal[:3]
            desired_pos = desired_goal[:3]
        else:
            achieved_pos = achieved_goal[:, :3]
            desired_pos = desired_goal[:, :3]

        new_contact = info["new_contact"]
        contact_release = info["contact_release"]
        contact_step = info["contact_step"]
        joint_velocities = info["joint_velocities:"]
        # print("joint velocities:", joint_velocities)
        d = distance(achieved_pos, desired_pos)

        if self.reward_type == "sparse":
            reward = -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            r_target = -d * 10.0
            r_new_contact = new_contact * 0.5
            r_contact_release = contact_release * 0.5
            r_contact_step = -contact_step * 0.5

            # success bonus
            r_success = np.where(
                d < self.distance_threshold,
                50.0,
                0.0
            )

            # penalty for orientation shift after success, encourage the agent to maintain the same orientation as much as possible
            r_orn_after_success = np.where(
                (d < self.distance_threshold) & np.any(np.abs(joint_velocities) > 0.01), # if the robot has large joint velocity after reaching the goal, consider it as orientation shift
                -20.0,
                0.0
            )

            # large contact penalty for safety exploring
            # if np.any(contact_force_case > 2.5): # if contact force is larger than 2.5N, consider it as contact is made
            #     r_large_contact = -10.0
            # else:
            #     r_large_contact = 0.0

            reward = r_target + r_new_contact + r_contact_release + r_contact_step + r_success + r_orn_after_success # + r_large_contact

        if np.isscalar(reward) or np.shape(reward) == ():
            return float(reward)
        return reward
        
    def angle_wrap(self, x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2 * np.pi) - np.pi

    def euler_error(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute wrapped per-axis error between two Euler-angle vectors.

        Args:
            a: current Euler angles, shape (3,)
            b: target Euler angles, shape (3,)

        Returns:
            Wrapped angle error in [-pi, pi], shape (3,)
        """
        assert a.shape == b.shape
        return self.angle_wrap(a - b)