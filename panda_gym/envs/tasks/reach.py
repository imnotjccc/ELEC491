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
        contact_flag=0,
        orientation_flag=0,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.contact_flag = contact_flag
        self.orientation_flag = orientation_flag

        self.goal_range_low = np.array([-0.10, -goal_range / 2, 0.10])
        self.goal_range_high = np.array([0.05, goal_range / 2, 0.45])
        # self.goal_range_high = np.array([0.05, 0.05, goal_range - 0.05])

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
        self.obsticle_name = "screen"
        # add some obsticles
        # add a screen
        # self.sim.create_box(
        #     body_name=self.obsticle_name,
        #     half_extents=np.array([0.01, 0.2, 0.2]),
        #     mass=0.0,
        #     position=np.zeros(10),
        #     rgba_color=np.array([0.1, 0.9, 0.1, 0.5]),
        # )

        # self.create_flag = random.uniform(0,1)
        # print(self.create_flag)
        # if(self.create_flag < 0.5):
        # create a new structure for pybulletx
        self.obj = px.Body(
            urdf_path="tacto/Obsticle_box/urdf/Obsticle_box.urdf",
            base_position=[-0.20, 0.0, 0.40],
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
        self.sim.create_sphere(
            body_name="red_goal",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.9, 0.1, 0.1, 0.4]),
        )

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
        self.red_goal = self.goal.copy()
        self.red_goal[0] -= 0.30 # make sure red goal is not
        # self.screen_pose = self.goal.copy()
        # self.screen_pose[0] -= 0.1

        self.sim.set_base_pose("red_goal", self.red_goal, np.array([0.0, 0.0, 0.0, 1.0]))
        # self.sim.set_base_pose("screen", self.screen_pose, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0])) # make target the new goal

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal
    
    def _sample_red_goal(self) -> np.ndarray:
        """Randomize second goal."""
        self.red_goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        self.red_goal[0] -= 0.10 # make sure red goal is not collide with original goal
        return self.red_goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    # def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
    #     # penalty for ee distance to taget
    #     d = distance(achieved_goal[:3], desired_goal[:3])

    #     if self.reward_type == "sparse":
    #         target_penalty = -np.array(d > self.distance_threshold, dtype=np.float64)
    #     else: # dense reward shaping
    #         target_penalty = -d

    #         achieved_orn = np.array(achieved_goal[3:], dtype=np.float64)
    #         desired_orn = np.array(desired_goal[3:], dtype=np.float64)

    #         orn_err = self.angle_wrap(achieved_orn - desired_orn)
    #         orn_penalty = -np.sum(orn_err ** 2, axis=-1)

    #         reward = target_penalty + 0.5 * orn_penalty

    #     # compute total reward
    #     reward = target_penalty

    #     return reward
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal = np.asarray(achieved_goal)
        desired_goal = np.asarray(desired_goal)

        if achieved_goal.ndim == 1:
            achieved_pos = achieved_goal[:3]
            achieved_orn = achieved_goal[3:6]
            desired_pos = desired_goal[:3]
            desired_orn = desired_goal[3:6]
            contact_force = achieved_goal[6:8]
            desire_contact_force = desired_goal[6:8]
        else:
            achieved_pos = achieved_goal[:, :3]
            achieved_orn = achieved_goal[:, 3:6]
            desired_pos = desired_goal[:, :3]
            desired_orn = desired_goal[:, 3:6]
            contact_force = achieved_goal[:, 6:8]
            desire_contact_force = desired_goal[:, 6:8]

        # print("contact force:", contact_force)
        d = distance(achieved_pos, desired_pos)
        contact_penalty = -distance(contact_force, desire_contact_force)

        if self.reward_type == "sparse":
            reward = -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            target_penalty = -d
            orn_err = self.angle_wrap(achieved_orn - desired_orn)
            orn_penalty = -np.sum(orn_err ** 2, axis=-1)
            reward = np.where(
                d < self.distance_threshold,
                target_penalty + orn_penalty * 0.01 + contact_penalty * 0.01, # add contact penalty for both cases
                target_penalty * 5.0 + contact_penalty * 0.05,
            )
        # print("target_penalty:", target_penalty, "orn_penalty:", orn_penalty, "contact_penalty:", contact_penalty)

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