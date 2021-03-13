
import random
from typing import Tuple
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState


class Actor:
    def __init__(self, discount_factor: float, decay_rate: float, learning_rate: float) -> None:
        self.discount_factor = discount_factor
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate

        self.policies = {}  # Pi(s)
        self.eligibilities = {}  # SAP-based eligibilities
        self.state = None

    def reset_eligibilities(self) -> None:
        self.eligibilities.clear()

    def set_eligibility(self, state: UniversalState, action: UniversalAction, value: int) -> None:
        self.eligibilities.setdefault(str(state), {})[str(action)] = value

    def generate_action(self, state: UniversalState, legal_actions: list, epsilon: float) -> UniversalAction:
        if random.uniform(0, 1) < epsilon:
            random_index = random.randint(0, len(legal_actions) - 1)
            chosen_action = legal_actions[random_index]

        else:
            action_policies = {}
            for action in legal_actions:
                self.policies.setdefault(str(state), {}).setdefault(str(action), 0)

                action_policies[action] = self.policies[str(state)][str(action)]

            chosen_action = max(action_policies, key=action_policies.get)

        universal_action = UniversalAction()
        universal_action.action = chosen_action
        return universal_action

    def compute_policies(self, td_error: float) -> None:
        for state_key, action_dict in self.eligibilities.items():
            for action_key, eligibility in action_dict.items():
                value = self.policies.setdefault(str(state_key), {}).setdefault(str(action_key), 0)

                self.policies[str(state_key)][str(action_key)] = value + self.learning_rate * eligibility * td_error

    def decay_eligibilities(self) -> None:
        for state_key, action_dict in self.eligibilities.items():
            for action_key, eligibility in action_dict.items():
                self.eligibilities[str(state_key)][str(action_key)] = self.decay_rate * self.discount_factor * eligibility
