import pickle
import numpy as np
from os import path, listdir


def get_data(prefix, data_folder):
    """
    Get data from all files in data_folder matching prefix and return as a numpy array

    Keyword arguments:
    prefix -- a string of the prefix to match
    data_folder -- location to fetch data from
    """
    files = [i for i in listdir(data_folder) if path.isfile(path.join(data_folder, i)) and prefix in i]
    files.sort()
    all_data = []
    for file in files:
        infile = open("data/" + file, "rb")
        data_in_file = pickle.load(infile)
        infile.close()
        for episode in data_in_file:
            all_data.append(episode)
    np_all_data = np.array(all_data)
    return np_all_data


def represent_actions(available_actions, all_actions):
    """
    Convert actions to a one-hot-encoded categorical vector

    Keyword arguments:
    available_actions -- all possible actions
    all_actions -- all the actions in the training data
    """
    actions = np.zeros([len(all_actions), len(available_actions)])
    for i, a in enumerate(available_actions):
        set_to = np.zeros(len(available_actions))
        set_to[i] = 1
        actions[np.all(all_actions == a, axis=1)] = set_to
    return actions


def drop_some_non_actions(states, actions, proportion_to_drop=0.25):
    """
    Drop some 0 action state action pairs

    The training data is fairly unbalanced with too many 0s, so this function drops a proportion
    (proportion_to_drop) of those

    Keyword arguments:
    states -- states in pre-collected data
    actions -- actions in pre-collected data. Dimension 0 of states and actions should match
    proportion_to_drop -- numerical argument between 0 and 1 indicating proportion of 0s to remove
    """
    assert states.shape[0] == actions.shape[0]
    no_action = np.argwhere(np.sum(np.abs(actions), axis=1) == 0).flatten()
    number_to_remove = round(len(no_action) * proportion_to_drop)
    to_remove = np.random.choice(no_action, number_to_remove)
    states = np.delete(states, to_remove, axis=0)
    actions = np.delete(actions, to_remove, axis=0)
    return states, actions
