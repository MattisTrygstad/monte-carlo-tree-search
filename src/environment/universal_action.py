class UniversalAction():
    """
    Superclass for representing an action any game. This contributes to decoupling of the environment and the agent.
    """

    def __init__(self, action: tuple = None) -> None:
        super().__init__()
        self.coordinates = action if action else ()

    def __str__(self) -> str:
        return str(self.coordinates)
