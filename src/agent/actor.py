
import random
from typing import Tuple
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


class Actor:
    def __init__(self, learning_rate: float, save_interval: int, input_neurons: int, nn_dimensions: list, activation_functions: list) -> None:
        self.learning_rate = learning_rate
        self.save_interval = save_interval

        self.replay_buffer = []  # [(PID, input, target),...]

        """ self.model = Sequential()
        self.model.add(Input(input_neurons))

        for x in range(nn_dimensions):
            self.model.add(Dense(nn_dimensions[x], activation_functions[x]))
        
        self.model.add(Dense(1)) """

    def generate_action(self, state: UniversalState, legal_actions: list, epsilon: float) -> UniversalAction:
        pass
