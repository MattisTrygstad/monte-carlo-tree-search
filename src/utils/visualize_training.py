

from matplotlib import pyplot as plt
import numpy as np


def visualize_training(losses: np.ndarray, accuracies: np.ndarray) -> None:
    plt.plot(losses, label='Loss')
    plt.plot(accuracies, label='Accuracy')
    # plt.xlabel('Minibatches')
    # plt.ylabel('Loss')
    plt.legend(loc="upper right")

    text = f"Title"
    plt.title(text)
    plt.show()
