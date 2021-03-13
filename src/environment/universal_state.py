
class UniversalState():
    """
    Superclass for representing the state in any game. This contributes to decoupling of the environment and the agent.
    """

    def __init__(self) -> None:
        super().__init__()
        self.nodes = {}

    def __str__(self) -> str:
        string_representation = ''
        for state in self.nodes.values():
            string_representation += str(state)

        return string_representation
