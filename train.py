import time

from load_data import train_data, test_data
import network as nw


# 初始化一个全新的神经网络
network = nw.NetWork([28 * 28, 16, 16, 10])
# 训练神经网络
start_time = time.time()
# 参数依次为：训练数据集、测试数据集、训练周期、每批样本量、学习速率
network.MBGD(train_data, test_data, 10, 1, 0.1)
end_time = time.time()
print(f"训练总用时:{end_time - start_time:.2f}秒")
# 可以将训练后的神经网络保存起来
network.save("checkpoints/model_28,16,16,10_epoch10_lr0.1_bs_1.pickle")
