from keras.layers import Dense, Flatten, Activation, MaxPooling2D, Conv2D
from keras.models import Sequential, save_model, load_model
from keras import Input
from keras.optimizers import Adam
from keras import metrics
import gym
from utils import rgb2grey, available_actions
from config import *
import numpy as np


def conv_nn(input_channels):
    """
    Build a convolutional neural network with 4 convolutional layers and 5 fully connected layers

    Keyword arguments
    input_channels -- the number of channels in the input
    """
    model = Sequential()

    model.add(Input(shape=(96, 96, input_channels)))

    # starts with four  convolutional and max-pooling layers
    model.add(Conv2D(filters=8, kernel_size=4, strides=(2, 2)))  # output shape (47, 47, 8)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))  # (46, 46, 8)

    model.add(Conv2D(filters=36, kernel_size=3, padding='same', strides=(2, 2)))  # (23, 23, 36)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))  # (22, 22, 36)

    model.add(Conv2D(filters=48, kernel_size=3, padding='same', strides=(2, 2)))  # (11, 11, 48)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))  # (10, 10, 48)

    model.add(Conv2D(filters=64, kernel_size=3, padding='same', strides=(2, 2)))  # (5, 5, 64)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))  # (4, 4, 64)

    model.add(Flatten())  # 1024

    # fully connected layers
    model.add(Dense(1024))  # 1024
    model.add(Activation('relu'))

    model.add(Dense(100))  # 100
    model.add(Activation('relu'))

    model.add(Dense(50))  # 50
    model.add(Activation('relu'))

    model.add(Dense(10))  # 10
    model.add(Activation('relu'))

    model.add(Dense(len(available_actions)))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(learning_rate),
                  loss=metrics.categorical_crossentropy,
                  metrics=metrics.categorical_accuracy,
                  )

    return model


def train(model, x_train, y_train, x_val, y_val, batch, epochs):
    """
    Train the model, and store the learning history

    Keyword argyments
    model -- a network
    batch -- the batch size
    epochs -- number of epochs to run the model for
    """
    history = model.fit(
                x_train,
                y_train,
                batch_size=batch,
                epochs=epochs,
                validation_data=(x_val, y_val),
            )

    return history, model


def agent_play(model):
    """
    View the agent's performance on the environment

    Keyword arguments
    model -- the trained agent
    """
    env_name = "CarRacing-v0"
    env = gym.make(env_name)

    env.render()

    is_open = True

    while is_open:
        first_state = env.reset()
        first_state_grey = rgb2grey(first_state)
        # to record velocity, we want to keep track of the X most recent states (X = number_states_to_keep)
        recent_states_fifo = [first_state_grey] * number_states_to_keep
        steps = 0
        total_reward = 0.0
        restart = False

        # loop over steps of an episode
        while True:
            state_for_model = np.transpose(recent_states_fifo, (1, 2, 0))
            state_for_model = np.array([state_for_model])
            a = model.predict(state_for_model)
            a = available_actions[a.argmax()]
            s, r, done, info = env.step(a)
            recent_states_fifo.pop(0)
            grey_state = rgb2grey(s)
            recent_states_fifo.append(grey_state)
            assert len(recent_states_fifo) == number_states_to_keep

            total_reward += r

            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))

            steps += 1
            is_open = env.render()
            if done or restart or is_open is False:
                break
    env.close()
