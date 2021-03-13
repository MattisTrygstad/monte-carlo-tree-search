

from abc import ABC, abstractmethod
from typing import Union

from environment.universal_state import UniversalState


class Approximator(ABC):
    """
    Abstract class to be implemented by each critic
    """

    def __init__(self, discount_factor) -> None:
        self.state_values = {}
        self.discount_factor = discount_factor
        self.eligibilities: Union[dict, list]

    @abstractmethod
    def compute_state_values(self, td_error: float, reinforcement: float, state: UniversalState, next_state: UniversalState) -> None:
        pass

    @abstractmethod
    def get_state_value(self, state: UniversalState) -> float:
        pass

    @abstractmethod
    def reset_eligibilities(self) -> None:
        pass

    @abstractmethod
    def set_eligibility(self, state: UniversalState, value: float) -> None:
        pass

    @abstractmethod
    def decay_eligibilies(self):
        pass
