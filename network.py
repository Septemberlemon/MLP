import numpy as np
import pickle


class NetWork:
    def __init__(self, sizes: list):
        self.layers_number = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.rand(x, y) - 0.5 for x, y in zip(sizes[1:], sizes[:-1])]
        self.biases = [np.zeros((x, 1)) for x in sizes[1:]]

    def MBGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            right_number = self.evaluate(test_data)
            print("初始正确率 {:.2f}%".format(right_number / len(test_data) * 100))

        n = len(training_data)
        for i in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size if k + mini_batch_size < n else n] for k in
                            range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.updateMiniBatch(mini_batch, eta)

            if test_data:
                right_number = self.evaluate(test_data)
                print("周期 {} 完成    正确率 {:.2f}%".format(i+1, right_number / len(test_data) * 100))

    def updateMiniBatch(self, mini_batch_data, eta):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        n = len(mini_batch_data)
        for sample in mini_batch_data:
            delta_nabla_w, delta_nabla_b = self.BP(sample)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [weight - (eta / n) * nabla_w for weight, nabla_w in zip(self.weights, nabla_w)]
        self.biases = [bias - (eta / n) * nabla_b for bias, nabla_b in zip(self.biases, nabla_b)]

    def BP(self, sample):
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        zs = []
        activations = [sample[0]]
        activation = sample[0]
        for weight, bias in zip(self.weights, self.biases):
            z = np.matmul(weight, activation) + bias
            zs.append(z)
            activation = NetWork.sigmoid(z)
            activations.append(activation)
        activations[-1][sample[1]] -= 1
        nabla_b[-1] = activations[-1] * NetWork.sigmoidDerivative(zs[-1])
        nabla_w[-1] = np.matmul(nabla_b[-1], activations[-2].transpose())
        for layer in range(2, self.layers_number):
            nabla_b[-layer] = np.matmul(self.weights[-layer + 1].T, nabla_b[-layer + 1]) * NetWork.sigmoidDerivative(
                zs[-layer])
            nabla_w[-layer] = np.matmul(nabla_b[-layer], activations[-layer - 1].T)

        return nabla_w, nabla_b

    def evaluate(self, evaluate_data):
        return sum(int(np.argmax(self.feedForward(x)) == y) for x, y in evaluate_data)

    def feedForward(self, input_data: np.array):
        output_data = input_data
        for weight, bias in zip(self.weights, self.biases):
            output_data = NetWork.sigmoid(np.matmul(weight, output_data) + bias)
        return output_data

    def save(self, file_name: str):
        parameters = (self.sizes, self.weights, self.biases)
        file = open("./" + file_name, "wb")
        pickle.dump(parameters, file)
        file.close()

    def load(self, file_name: str):
        file = open("./" + file_name, "rb")
        self.sizes, self.weights, self.biases = pickle.load(file)
        self.layers_number = len(self.sizes) + 1
        file.close()

    @staticmethod
    def sigmoid(input_vector: np.array):
        return 1 / (1 + np.exp(-input_vector))

    @staticmethod
    def sigmoidDerivative(input_vector: np.array):
        return NetWork.sigmoid(input_vector) * (1 - NetWork.sigmoid(input_vector))
