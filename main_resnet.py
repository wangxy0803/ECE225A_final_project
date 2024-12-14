import torchvision.models as models
import torch
import torch.nn as nn
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt



# 导入数据
print("downloading cluster data...")
data = loadmat('./data_0104/cluster6_kmeans.mat')
dataset = data['cluster6']


# 按行（axis = 0）去除重复数据，按照行排序好
dataset = np.unique(dataset, axis=0)

# 划分数据集并缩放数据
# random_state参数控制随机种子
X_train, X_test, y_train, y_test = train_test_split(dataset[:, 0:3], dataset[:, 3:9], test_size=0.2)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)


# 缩放数据【归一化】
scaler = MinMaxScaler(feature_range=(-1, 1))
input_train_scaled = scaler.fit_transform(X_train)
output_train_scaled = scaler.fit_transform(y_train)

input_test_scaled = scaler.fit_transform(X_test)
output_test_scaled = scaler.fit_transform(y_test)

# input_val_scaled = scaler.fit_transform(X_val)
# output_val_scaled = scaler.fit_transform(y_val)


# 将numpy数组转成torch的张量
input_train_tensor = torch.from_numpy(input_train_scaled).float()
output_train_tensor = torch.from_numpy(output_train_scaled).float()
# input_val_tensor = torch.from_numpy(input_val_scaled).float()
# output_val_tensor = torch.from_numpy(output_val_scaled).float()
input_test_tensor = torch.from_numpy(input_test_scaled).float()
output_test_tensor = torch.from_numpy(output_test_scaled).float()



# 载入网络模型
resnet18 = models.resnet18()

# 修改网络层的channel
# print(resnet18.conv1)
resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


# 修改全连接层的输出
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 6)


# 定义优化器
# optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.5, momentum=0.9, weight_decay=0.001)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.5, momentum=0.99)
optimizer = torch.optim.RMSprop(resnet18.parameters(), lr=0.001, alpha=0.9, eps=1e-08, weight_decay=0, momentum=0, centered=False)
# optimizer = torch.optim.NAdam(net.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004, foreach=None)
# optimizer = torch.optim.AdamW(net.parameters(), lr=0.5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = torch.optim.Adagrad(net.parameters(), lr=0.1)


# 定义损失函数
loss_func = nn.MSELoss()


# GPU训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet18 = resnet18.to(device)
# resnet18.eval()
loss_func = loss_func.to(device)


# 画图准备
x = []                  # 横坐标
train_loss_list = []    # train损失值
val_loss_list = []      # val损失值


# GPU训练
input_train_tensor = input_train_tensor.to(device)
output_train_tensor = output_train_tensor.to(device)

# input_val_tensor = input_val_tensor.to(device)
# output_val_tensor = output_val_tensor.to(device)

max_epoch = 10000
# batch_size = 6000
# numInterationsPerEpoch = len(input_train_tensor[0, 0, :, :]) // batch_size

# print(input_train_tensor.shape)
# 转化成B×C×H×W
input_train_tensor.unsqueeze_(1)
input_train_tensor.unsqueeze_(2)
# print(input_train_tensor.shape)

for epoch in range(max_epoch):
    # whole_data_index = list(range(len(input_train_tensor[0, 0, :, :])))
    # random.shuffle(whole_data_index)
    
    prediction = resnet18(input_train_tensor)    # 前向传播
    # print(prediction.shape)
    loss = loss_func(prediction, output_train_tensor)   # 计算误差
    optimizer.zero_grad()   # 为下面的反向传播清除梯度（否则梯度会叠加）
    loss.backward()         # 反向传播计算梯度
    optimizer.step()   

    if epoch % 10 == 0:
        x.append(epoch)
        train_loss_list.append(loss.data.cpu().numpy())
        # val_loss_list.append(torch.sqrt(val_loss))

        # 绘制训练过程
        # 允许动态画图
        plt.ion()
        plt.cla()   # 清除坐标轴
        # plt.scatter(input_train_tensor.data.numpy(), output_train_tensor.data.numpy())
        # plt.plot(input_train_tensor.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        # plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        try:
            train_loss_lines.remove(train_loss_lines[0])    # 移除上一步曲线
        except Exception:
            pass

        train_loss_lines = plt.plot(x, train_loss_list, 'r', lw=1)  # lw为曲线宽度
        plt.title("loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(["train_loss"])
        plt.pause(0.1)
        
    if epoch % 3000 == 0:
        x = []
        train_loss_list = []

    if epoch == max_epoch - 1:
        # 取出预测数据与label数据并化成numpy形式
        prediction = resnet18(input_train_tensor)    # 前向传播
        prediction = prediction.data.cpu().numpy()
        output_train_tensor = output_train_tensor.data.cpu().numpy()

        # 反归一化
        prediction = scaler.inverse_transform(prediction)
        output_train_tensor = scaler.inverse_transform(output_train_tensor)
        
        # 把预测数据与label数据化成torch的tenser形式
        prediction = torch.from_numpy(prediction).float()
        output_train_tensor = torch.from_numpy(output_train_tensor).float()
        
        loss_angle = loss_func(prediction[:, 0:3], output_train_tensor[:, 0:3])
        print('角度预测RMSE:')
        # print(torch.sqrt(loss_angle) / math.pi * 180)
        print(torch.sqrt(loss_angle))
        # print('例子:')
        # print(prediction[0, 0:3], output_train_tensor[0, 0:3])

        loss_position = loss_func(prediction[:, 3:6], output_train_tensor[:, 3:6])
        print('位移预测RMSE:')
        # print(torch.sqrt(loss_position) * distance_scale)
        print(torch.sqrt(loss_position))

    # for batch in range(numInterationsPerEpoch):
    #     batch_index = whole_data_index[batch*batch_size:(batch+1)*batch_size]
        # print(2)
        # prediction = resnet18(input_train_tensor[:, :, batch_index, :])    # 前向传播
        # print(prediction.shape)
        # loss = loss_func(prediction, output_train_tensor[batch_index, :])   # 计算误差
        # optimizer.zero_grad()   # 为下面的反向传播清除梯度（否则梯度会叠加）
        # loss.backward()         # 反向传播计算梯度
        # optimizer.step()        # 利用计算的梯度代入优化器，继而更新参数

plt.ioff()
plt.show()
plt.savefig('train_loss')
plt.close()

# 把数据放入GPU中
input_test_tensor = input_test_tensor.to(device)

input_test_tensor.unsqueeze_(1)
input_test_tensor.unsqueeze_(2)

# 测试网络模型
test_prediction = resnet18(input_test_tensor)

test_prediction = test_prediction.data.cpu().numpy()
output_test_tensor = output_test_tensor.data.cpu().numpy()

# 反归一化
test_prediction = scaler.inverse_transform(test_prediction)
output_test_tensor = scaler.inverse_transform(output_test_tensor)

# 把预测数据与label数据化成torch的tenser形式
test_prediction = torch.from_numpy(test_prediction).float()
output_test_tensor = torch.from_numpy(output_test_tensor).float()

print("********************************************************************")
print("测试集的角度预测RMSE:")
# print(torch.sqrt(loss_func(test_prediction[:, 0:3], output_test_tensor[:, 0:3])).data.cpu() / math.pi * 180)
print(torch.sqrt(loss_func(test_prediction[:, 0:3], output_test_tensor[:, 0:3])).data.cpu())
print("测试集的位移预测RMSE:")
# print(torch.sqrt(loss_func(test_prediction[:, 3:6], output_test_tensor[:, 3:6])).data.cpu() * distance_scale)
print(torch.sqrt(loss_func(test_prediction[:, 3:6], output_test_tensor[:, 3:6])).data.cpu())


# 保存网络模型结构以及参数
torch.save(resnet18, 'resnet18_cluster1_1225_2.pkl')

# # 取出模型
# net_record = torch.load('net.pkl')



