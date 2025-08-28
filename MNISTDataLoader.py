import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from struct import unpack
from array import array


class MNISTDataLoader :
    def __init__(
            self,
            training_images_filepath,
            training_labels_filepath,
            test_images_filepath,
            test_labels_filepath
        ) :

        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None

        self.load_images()

    def read_images_labels(self, images_filepath, labels_filepath) :
        labels = []
        with open(labels_filepath, 'rb') as file :
            magic, size = unpack(">II", file.read(8))
            if magic != 2049 :
                raise ValueError(f"Magic number mismatch, expected 2049, got {magic}")

            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file :
            magic, size, rows, cols = unpack(">IIII", file.read(16))
            if magic != 2051 :
                raise ValueError(f"Magic number mismatch, expected 2051, got {magic}")

            image_data = array("B", file.read())

        images = []
        for i in range(size) :
            images.append([0] * rows * cols)
        for i in range(size) :
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_images(self) :
        self._x_train, self._y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        self._x_test, self._y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)

    @property
    def x_train(self) :
        if self._x_train is None :
            raise ValueError("Training data not loaded, call load_images()")
        return self._x_train

    @property
    def y_train(self) :
        if self._y_train is None :
            raise ValueError("Training data not loaded, call load_images()")
        return self._y_train

    @property
    def x_test(self) :
        if self._x_test is None :
            raise ValueError("Testing data not loaded, call load_images()")
        return self._x_test

    @property
    def y_test(self) :
        if self._y_test is None :
            raise ValueError("Testing data not loaded, call load_images()")
        return self._y_test

    def show_images(self, figsize=(12,8)) :
        num_images = 15
        rows, cols = 3, 5

        if self._x_train is None or self._x_test is None :
            raise ValueError("No data loaded, call load_images() ")

        all_images = self._x_train + self._x_test
        all_labels = self._y_train + self._y_test

        indices = np.random.choice(len(all_images), num_images, replace=False)

        images = [all_images[i] for i in indices]
        titles = [f"Image [{i}] = {all_labels[i]}" for i in indices]

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()

        for i, (x, y) in enumerate(zip(images, titles)) :
            ax = axes[i]
            ax.imshow(x, cmap="gray")
            ax.set_title(y, fontsize=10)
            ax.axis("off")

        for j in range(i + 1, len(axes)) :
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()


home = Path.home()
input_path = home / "Desktop" / "Pycharm Projects" / "MNIST Python" / "archive"
training_image_filepath = input_path / "train-images-idx3-ubyte" / "train-images-idx3-ubyte"
training_label_filepath = input_path / "train-labels-idx1-ubyte" / "train-labels-idx1-ubyte"
test_image_filepath = input_path / "t10k-images-idx3-ubyte" / "t10k-images-idx3-ubyte"
test_label_filepath = input_path / "t10k-labels-idx1-ubyte" / "t10k-labels-idx1-ubyte"

mnist_dataloader = MNISTDataLoader(
    training_image_filepath,
    training_label_filepath,
    test_image_filepath,
    test_label_filepath
)

mnist_dataloader.show_images()
