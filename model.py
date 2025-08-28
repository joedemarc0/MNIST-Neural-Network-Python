import numpy as np
from MNISTDataLoader import MNISTDataLoader


class NeuralNetwork :
    def __init__(self) :
        self.W1 = np.random.randn(128, 784) * np.sqrt(2 / 784)
        self.W2 = np.random.randn(128, 64) * np.sqrt(2 / 128)
        self.W3 = np.random.randn(10, 64) * np.sqrt(2 / 64)

        self.b1 = np.zeros((128, 1))
        self.b2 = np.zeros((64, 1))
        self.b3 = np.zeros((10, 1))

    @staticmethod
    def leakyReLU(x, alpha=0.01) :
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def softmax(x) :
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    @staticmethod
    def oneHot(y) :
        one_hot_y = np.zeros((10, y.size))
        one_hot_y[y, np.arange(y.size)] = 1
        return one_hot_y

    @staticmethod
    def derivLeakyReLU(x, alpha=0.01) :
        return np.where(x > 0, 1, alpha)

    @staticmethod
    def computeLoss(y_true, y_pred) :
        m = y_true.shape[1]
        epsilon = 1e-12
        clipped_preds = np.clip(y_pred, epsilon, 1. - epsilon)
        loss = - np.sum(y_true * np.log(clipped_preds)) / m
        return loss

    def forwardProp(self, x) :
        Z1 = self.W1.dot(x) + self.b1
        A1 = self.leakyReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.leakyReLU(Z2)
        Z3 = self.W3.dot(A2) + self.b3
        A3 = self.softmax(Z3)
        return Z1, A1, Z2, A2, Z3, A3

    def backProp(self, x, y, Z1, A1, Z2, A2, Z3, A3) :
        m = y.size
        one_hot_y = self.oneHot(y)

        dZ3 = A3 - one_hot_y
        dW3 = dZ3.dot(A2.T) / m
        db3 = np.sum(dZ3, axis=1, keepdims=True) / m

        dZ2 = self.W3.T.dot(dZ3) * self.derivLeakyReLU(Z2)
        dW2 = dZ2.dot(A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = self.W2.T.dot(dZ2) * self.derivLeakyReLU(Z1)
        dW1 = dZ1.dot(x.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        return dW1, db1, dW2, db2, dW3, db3

    def update_params(self, dW1, db1, dW2, db2, dW3, db3, eta) :
        self.W1 -= eta * dW1
        self.b1 -= eta * db1
        self.W2 -= eta * dW2
        self.b2 -= eta * db2
        self.W3 -= eta * dW3
        self.b3 -= eta * db3

    @staticmethod
    def get_predictions(A) :
        return np.argmax(A, 0)

    @staticmethod
    def get_accuracy(predictions, y) :
        return np.mean(predictions == y)

    @staticmethod
    def get_batches(x, y, batch_size=64) :
        m = x.shape[1]
        permutation = np.random.permutation(m)
        x_shuffled = x[:, permutation]
        y_shuffled = y[permutation]

        batches = []
        for k in range(0, m, batch_size) :
            batch_x = x_shuffled[:, k:k+batch_size]
            batch_y = y_shuffled[k:k+batch_size]
            batches.append((batch_x, batch_y))
            yield batch_x, batch_y

    def train(
            self,
            x,
            y,
            x_val,
            y_val,
            num_iter=100,
            init_eta=0.1,
            batch_size=64,
            decay_rate=0.99
        ):

        for i in range(num_iter) :
            eta = init_eta * (decay_rate ** i)

            for batch_x, batch_y in self.get_batches(x, y, batch_size=batch_size) :
                Z1, A1, Z2, A2, Z3, A3 = self.forwardProp(batch_x)
                dW1, db1, dW2, db2, dW3, db3 = self.backProp(batch_x, batch_y, Z1, A1, Z2, A2, Z3, A3)
                self.update_params(dW1, db1, dW2, db2, dW3, db3, eta)

            if i % 10 == 0 or i == num_iter - 1 :
                _, _, _, _, _, A3_train = self.forwardProp(x)
                preds_train = self.get_predictions(A3_train)
                acc_train = self.get_accuracy(preds_train, y)

                if x_val is not None and y_val is not None :
                    _, _, _, _, _, A3_val = self.forwardProp(x_val)
                    preds_val = self.get_predictions(A3_val)
                    acc_val = self.get_accuracy(preds_val, y_val)
                    print(f"Iteration {i:03}: Train Accuracy = {acc_train:.4f} | Validation Accuracy = {acc_val:.4f} | Learning Rate = {eta:.5f}")
                else :
                    print(f"Iteration {i:03}: Train Accuracy = {acc_train:.4f} | Learning Rate = {eta:.5f}")

        print(f"Training Complete after {num_iter} iterations")

    def test_set(self, x_test, y_test) :
        _, _, _, _, _, A3_test = self.forwardProp(x_test)
        preds_test = self.get_predictions(A3_test)
        acc_test = self.get_accuracy(preds_test, y_test)
        print(f"Accuracy on test set = {acc_test:.4f}")
