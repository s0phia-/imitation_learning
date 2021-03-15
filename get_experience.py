import gym
import random
import numpy as np
import pickle
import time
from config import number_states_to_keep, data_folder
from utils import rgb2grey
import copy

env_name = "CarRacing-v0"
env = gym.make(env_name)


if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    is_open = True
    # loop over episodes
    while is_open:
        first_state = env.reset()
        # convert colour image to greyscale
        first_state_grey = rgb2grey(first_state)
        # to record velocity, we want to keep track of the X most recent states (X = number_states_to_keep)
        recent_states_fifo = [first_state_grey] * number_states_to_keep
        total_reward = 0.0
        steps = 0
        restart = False
        # log visited states and actions
        state_log = []
        action_log = []
        # loop over steps of an episode
        while True:
            s, r, done, info = env.step(a)
            # add the visited action to the action log
            action_log.append(copy.deepcopy(a))
            # keep a fifo list of the most recently visited states
            recent_states_fifo.pop(0)
            grey_state = rgb2grey(s)
            recent_states_fifo.append(grey_state)
            assert len(recent_states_fifo) == number_states_to_keep
            # add the stack of recently visited states to the state log
            state_log.append(recent_states_fifo)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            is_open = env.render()
            if done or restart or is_open is False:
                # save a pickle of the visited states and actions
                timestamp = time.strftime('%b-%d-%Y_%H%M%S', time.localtime())
                for i in [["states", state_log], ["actions", action_log]]:
                    filename = data_folder + i[0] + '_' + timestamp + ".pickle"
                    outfile = open(filename, 'wb')
                    pickle.dump(i[1], outfile)
                    outfile.close()
                break
    env.close()
