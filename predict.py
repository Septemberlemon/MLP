import random

import numpy as np
import matplotlib.pyplot as plt

from load_data import train_data, test_data
import network as nw


network = nw.NetWork([])  # 先创建一个空的神经网络
network.load("checkpoints/model_28,16,16,10_epoch10_lr0.1_bs_1.pickle")  # 再加载一个已经训练好的模型重新初始化该神经网络
# 看一下这个模型的正确率
len_train = len(train_data)
len_test = len(test_data)
num_correct_test = network.evaluate(test_data)
num_correct_train = network.evaluate(train_data)
print(f"训练集正确率{num_correct_train / len_train * 100:.2f}%    "
      f" 测试集正确率 {num_correct_test / len_test * 100:.2f}%")

# 随机选取一个图片进行预测
img, label = test_data[random.randint(0, len_test - 1)]
predicted_label = np.argmax(network.feedForward(img))

# 展示预测结果
plt.imshow(img.reshape(28, 28), cmap="gray")
plt.title(f"Label: {label}\nPrediction: {predicted_label}")
plt.show()
