import time

import network
import load_data

# 初始化一个全新的神经网络
# network = network.NetWork([28*28, 16, 16, 10])

# 或者直接使用已经训练好的参数
network = network.NetWork([])  # 先创建一个空的神经网络
network.load("parameter.pickle")  # 再加载一个已经训练好的模型重新初始化该神经网络

# 训练神经网络
# start_time = time.time()
# # 参数依次为：训练数据集、训练周期、小批量数据大小、学习速率、测试数据
# network.MBGD(load_data.training_data, 5, 1, 1, load_data.test_data)
# end_time = time.time()
# print("训练总用时:{:.2f}秒".format(end_time - start_time))

# 将训练后的神经网络保存起来
# network.save("parameter.pickle")

# 加载测试数据对网络效果进行验证
n = len(load_data.training_data)  # 获取测试数据样本量
num = network.evaluate(load_data.training_data)  # 获取正确样本量
# 打印结果
print("模型正确率：{}/{} {:.2f}%".format(num, n, num / n * 100))
