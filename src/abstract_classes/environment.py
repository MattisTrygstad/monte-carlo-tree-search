from abc import ABC, abstractmethod
from typing import Tuple

from environment.universal_state import UniversalState


class Environment(ABC):
    """
    Abstract class to be implemented by each environment
    required tables.
    """

    @abstractmethod
    def execute_action(self, action: tuple) -> None:
        pass

    @abstractmethod
    def get_legal_actions(self) -> list:
        pass

    @abstractmethod
    def check_win_condition(self) -> bool:
        pass

    @abstractmethod
    def get_state(self) -> UniversalState:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def visualize(self) -> None:
        pass
