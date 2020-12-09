import metaworld
import random
import time
import numpy as np
import metaworld.envs.mujoco.env_dict as _env_dict


def run(env, task_iter):
    # Runs an interactive GUI
    task = next(task_iter)
    env.set_task(task)
    env.max_path_length = 1e10 # Disable max episode length
    obs = env.reset()
    env.render(mode='human')
    viewer = env.viewer
    step = 0
    while True:
        step += 1
        env.render(mode='human')
        a = viewer.get_user_action()
        obs, reward, done, info = env.step(a)
        time.sleep(.075)
        display_dict = {
            'Task Name': task.env_name,
            'Step': step,
            'Success': info['success'],
            'EpisodeRew': info['epRew'],
        }
        for idx, obs_feat in enumerate(obs):
            display_dict["obs_{}".format(idx)] = obs_feat
        viewer.display_info(display_dict)
        if viewer.newtask_requested:
            viewer.newtask_requested = False
            break
        if done or env.curr_path_length >= env.max_path_length or viewer.reset_requested:
            task = next(task_iter)
            env.set_task(task)
            env.reset()
            viewer.reset_requested = False
            step = 0
    env.close()


if __name__ == "__main__":
    v2_envs = list(metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS.keys())
    random.shuffle(v2_envs)
    for task_name in v2_envs:
        ml1 = metaworld.V2(task_name)
        env = ml1.train_classes[task_name]()
        run(env, iter(ml1.train_tasks))