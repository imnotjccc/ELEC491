from typing import Any, Dict, Union

import numpy as np

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
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.contact_flag = contact_flag

        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0.15])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range - 0.05])

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

        # add some obsticles
        # add a screen
        # self.sim.create_box(
        #     body_name="screen",
        #     half_extents=np.array([0.01, 0.2, 0.1]),
        #     mass=0.0,
        #     position=np.zeros(3),
        #     rgba_color=np.array([0.1, 0.9, 0.1, 0.5]),
        # )

        #new goal poaition (red)
        self.sim.create_sphere(
            body_name="target_2",
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
        # if self.contact_flag == 0:
        # add a screen obstacle between the robot and the goal
        # self.screen_goal = self._sample_goal()
        # self.sim.set_base_pose("screen", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

        # self.goal[0] -= 0.10 # generate goal in front of the screen
        self.goal[0] -= 0.10
        self.sim.set_base_pose("target_2", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.red_goal = self.goal.copy() # store the red goal position
        self.goal[0] += 0.10 # generate goal behind the screen
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0])) # make target the new goal
        # else:
        #     self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        #     self.green_goal = self.goal.copy() # store the green goal position
        #     self.goal[0] -= 0.10 # generate goal in front of the screen
        #     self.sim.set_base_pose("target_2", self.goal, np.array([0.0, 0.0, 0.0, 1.0])) # make target2 the new goal

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal
    
    # def _sample_second_goal(self) -> np.ndarray:
    #     """Randomize second goal."""
    #     goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
    #     goal[0] += 0.10 # generate goal behind the screen
    #     return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        # penalty for ee distance to taget
        d = distance(achieved_goal, desired_goal)

        if self.reward_type == "sparse":
            target_penalty = -np.array(d > self.distance_threshold, dtype=np.float64)
        else: # dense reward shaping
            target_penalty = -d

        # compute total reward
        reward = target_penalty

        return reward