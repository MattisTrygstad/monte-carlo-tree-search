
from abstract_classes.approximator import Approximator
from agent.table_approximator import TableApproximator
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState


class Critic:

    def __init__(self, approximator: Approximator) -> None:
        self.approximator = approximator  # Table or NN

    def compute_temporal_difference_error(self, state: UniversalState, next_state: UniversalAction, reinforcement: int) -> float:
        state_value = self.approximator.get_state_value(state)
        next_state_value = self.approximator.get_state_value(next_state)

        return reinforcement + self.approximator.discount_factor * next_state_value - state_value

    def compute_state_values(self, td_error: float, reinforcement: float, state: UniversalState, next_state: UniversalState) -> None:
        self.approximator.compute_state_values(td_error, reinforcement, state, next_state)

    def initialize_state_value(self, state: UniversalState) -> None:
        self.approximator.initialize_state_value(state)

    def reset_eligibilities(self) -> None:
        self.approximator.reset_eligibilities()

    def set_eligibility(self, state: UniversalState, value: int):
        self.approximator.set_eligibility(state, value)

    def decay_eligibilities(self) -> None:
        self.approximator.decay_eligibilies()
