import numpy as np
import pickle


class NetWork:
    def __init__(self, sizes: list):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]

    def MBGD(self, train_data, test_data, epochs, batch_size, learning_rate):
        len_train = len(train_data)
        len_test = len(test_data)
        num_correct_test = self.evaluate(test_data)
        num_correct_train = self.evaluate(train_data)
        print("测试集初始正确率 {:.2f}%".format(num_correct_test / len_test * 100))
        print("训练集初始正确率 {:.2f}%".format(num_correct_train / len_train * 100))

        for epoch in range(epochs):
            np.random.shuffle(train_data)
            batches = [train_data[k:k + batch_size] for k in range(0, len(train_data), batch_size)]
            for batch in batches:
                self.updateBatch(batch, learning_rate)

            num_correct_train = self.evaluate(train_data)
            num_correct_test = self.evaluate(test_data)
            print(f"周期 {epoch + 1} 完成    "
                  f"训练集正确率{num_correct_train / len_train * 100:.2f}%    "
                  f" 测试集正确率 {num_correct_test / len_test * 100:.2f}%")

    def updateBatch(self, batch, learning_rate):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        batch_size = len(batch)
        for sample in batch:
            delta_nabla_w, delta_nabla_b = self.BP(sample)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [weight - learning_rate / batch_size * nabla_w for weight, nabla_w in zip(self.weights, nabla_w)]
        self.biases = [bias - learning_rate / batch_size * nabla_b for bias, nabla_b in zip(self.biases, nabla_b)]

    def BP(self, sample):
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]

        zs, activations = [], [sample[0]]
        activation = sample[0]
        for weight, bias in zip(self.weights, self.biases):
            z = np.matmul(weight, activation) + bias
            zs.append(z)
            activation = NetWork.sigmoid(z)
            activations.append(activation)

        activations[-1][sample[1]] -= 1
        # loss = np.sum([0.5 * loss ** 2 for loss in activations])
        nabla_b[-1] = activations[-1] * NetWork.sigmoidDerivative(zs[-1])
        nabla_w[-1] = np.matmul(nabla_b[-1], activations[-2].T)
        for layer in range(2, self.num_layers):
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
        file = open(file_name, "wb")
        pickle.dump(parameters, file)
        file.close()

    def load(self, file_name: str):
        file = open(file_name, "rb")
        self.sizes, self.weights, self.biases = pickle.load(file)
        self.num_layers = len(self.sizes) + 1
        file.close()

    @staticmethod
    def sigmoid(input_vector: np.array):
        return 1 / (1 + np.exp(-input_vector))

    @staticmethod
    def sigmoidDerivative(input_vector: np.array):
        return NetWork.sigmoid(input_vector) * (1 - NetWork.sigmoid(input_vector))
