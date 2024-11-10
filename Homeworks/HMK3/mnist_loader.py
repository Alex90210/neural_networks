import numpy as np
from torchvision.datasets import MNIST

def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)

    mnist_data = []
    mnist_labels = []
    for idx, (image, label) in enumerate(dataset):
        mnist_data.append(image)
        mnist_labels.append(label)

    mnist_data = np.array(mnist_data)
    mnist_labels = np.array(mnist_labels)

    return mnist_data, mnist_labels