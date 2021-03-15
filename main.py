import numpy as np
from agent import conv_nn, train, agent_play, save_model, load_model
from data_preparation import get_data, represent_actions, drop_some_non_actions
from config import number_states_to_keep, number_of_epochs, batch_size, data_folder
from sklearn.model_selection import train_test_split
from utils import available_actions
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # collect expert data
    actions = get_data("actions", data_folder)
    states = get_data("states", data_folder)
    # remove some 0 actions to balance dataset
    states, actions = drop_some_non_actions(states, actions)
    # convert actions to one-hot encoded categorical vector
    actions = represent_actions(available_actions, actions)
    # get states in shape for CNN by switching axis to put channel last
    states = np.transpose(states, (0, 2, 3, 1))
    # split training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(states,
                                                      actions,
                                                      test_size=0.33,
                                                      shuffle=True)
    # build model
    model = conv_nn(number_states_to_keep)
    # fit model
    history, model = train(model, X_train, y_train, X_val, y_val, batch_size, number_of_epochs)
    # save model
    save_model(model, "saved_model")
    # model = load_model("saved_model")

    # Plot history
    plt.plot(history.history['loss'], label='Categorical Crossentropy Loss (training data)')
    plt.plot(history.history['val_loss'], label='Categorical Crossentropy Loss (validation data)')
    plt.title('Categorical Crossentropy Loss for CarRacing-v0')
    plt.ylabel('Categorical Crossentropy Loss value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()
    plt.savefig('foo.png')

    # view the agent playing the game
    agent_play(model)
