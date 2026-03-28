import os

from gym.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

# for reward_type in ["sparse", "dense"]:
#     for control_type in ["ee", "joints", "orn"]:
#         reward_suffix = "Dense" if reward_type == "dense" else ""
#         control_suffix = "Joints" if control_type == "joints" else ""
#         kwargs = {"reward_type": reward_type, "control_type": control_type}
#         # print(control_type)

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints", "orn"]:
        reward_suffix = "Dense" if reward_type == "dense" else ""
        if control_type == "joints":
            control_suffix = "Joints"
            # print("joint")
        elif control_type == "orn":
            control_suffix = "Orn"  # Add unique suffix for "orn"
            # print("orn")
        else:  # "ee"
            control_suffix = ""
            # print("ee")
        kwargs = {"reward_type": reward_type, "control_type": control_type}
        # print(control_type)

        register(
            id="PandaReach{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPush{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaPushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaSlide{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaSlideEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPickAndPlace{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaStack{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaStackEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id="PandaFlip{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaFlipEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )
