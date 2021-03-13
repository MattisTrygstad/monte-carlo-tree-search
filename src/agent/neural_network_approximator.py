
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow import GradientTape
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.optimizer_v2.adagrad import Adagrad
from abstract_classes.approximator import Approximator
from environment.universal_state import UniversalState
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


class NeuralNetworkApproximator(Approximator):

    def __init__(self, inputSize: int, nn_dimentions: list, activation_functions: list, learning_rate: float, discount_factor: float, decay_rate: float) -> None:
        super().__init__(discount_factor)

        self.eligibilities = []

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.decay_rate = decay_rate

        self.model = Sequential()
        self.model.add(Input(inputSize))

        for x in range(len(nn_dimentions)):
            self.model.add(Dense(nn_dimentions[x], activation=activation_functions[x]))

        self.model.add(Dense(1))
        self.reset_eligibilities()

        optimizer = Adagrad(learning_rate=learning_rate)
        loss_function = MeanSquaredError()
        self.model.compile(optimizer, loss_function, run_eagerly=True)

    def compute_state_values(self, td_error: float, reinforcement: float, state: UniversalState, next_state: UniversalState) -> None:
        with GradientTape() as gradient_tape:
            state, next_state, discount_factor, reinforcement = NeuralNetworkApproximator.__convert_to_tensors(state, next_state, self.discount_factor, reinforcement)

            target_value = tf.add(reinforcement, tf.multiply(discount_factor, self.model(next_state)))

            predicted_value = self.model(state)
            loss = self.model.loss(target_value, predicted_value)

        gradients = gradient_tape.gradient(loss, self.model.trainable_variables)

        updated_gradients = self.__customize_gradients(gradients, td_error)

        self.model.optimizer.apply_gradients(zip(updated_gradients, self.model.trainable_variables))

    def get_state_value(self, state: UniversalState) -> float:
        state_array = [tf.strings.to_number(value, out_type=tf.dtypes.int32) for value in str(state)]
        state_tensor = tf.convert_to_tensor(np.expand_dims(state_array, axis=0))
        return self.model(state_tensor).numpy()[0][0]

    def __convert_to_tensors(state: UniversalState, next_state: UniversalState, discount_rate: float, reinforcement: float) -> tuple:
        state_array = [tf.strings.to_number(value, out_type=tf.dtypes.float32) for value in str(state)]
        state_tensor = tf.convert_to_tensor(np.expand_dims(state_array, axis=0))

        next_state_array = [tf.strings.to_number(value, out_type=tf.dtypes.float32) for value in str(next_state)]
        next_state_tensor = tf.convert_to_tensor(np.expand_dims(next_state_array, axis=0))

        discount_rate_tensor = tf.convert_to_tensor(discount_rate, dtype=tf.dtypes.float32)
        reinforcement_tensor = tf.convert_to_tensor(reinforcement, dtype=tf.dtypes.float32)

        return state_tensor, next_state_tensor, discount_rate_tensor, reinforcement_tensor

    def __customize_gradients(self, gradients: defaultdict, td_error: float) -> tuple:
        for index in range(len(gradients)):
            if td_error == 0:
                break

            gradients[index] *= 0.5 / td_error
            self.eligibilities[index] += gradients[index]
            gradients[index] = self.eligibilities[index] * td_error

        return gradients

    def reset_eligibilities(self) -> None:
        self.eligibilities.clear()
        for var in self.model.trainable_variables:
            self.eligibilities.append(tf.zeros_like(var))

    def set_eligibility(self, state: UniversalState, value: float) -> None:
        pass

    def decay_eligibilies(self) -> None:
        for i in range(len(self.eligibilities)):
            self.eligibilities[i] = self.decay_rate * self.discount_factor * self.eligibilities[i]
