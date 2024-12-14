import torch.nn as nn 
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import torch
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pyswarms as ps
from sklearn.model_selection import train_test_split
import math
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
import torch.nn.functional as F

from CNN1DNetworkMO import CNN1Dmo

from GNN import GraphConvolutionLayer,GNN


# 导入数据
print("downloading cluster data...")
data = loadmat('./data/data_0319/cluster1_kmeans.mat')
dataset = data['cluster1']

# 把距离单位改成m，把角度单位改成弧度
# dataset[:, 0:3] = dataset[:, 0:3] /180 * math.pi
# dataset[:, 3:6] = dataset[:, 3:6] /180 * math.pi
# dataset[:, 6:9] = dataset[:, 6:9] /100

# 按行（axis = 0）去除重复数据，按照行排序好
dataset = np.unique(dataset, axis=0)

# 划分数据集并缩放数据
# random_state=20控制随机种子
X_train, X_test, y_train, y_test = train_test_split(dataset[:, 0:3], dataset[:, 3:9], test_size=0.2, random_state=20)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=20)

# # 缩放数据【归一化】
# scaler = MinMaxScaler(feature_range=(0, 1))
# input_train_scaled = scaler.fit_transform(X_train)
# # 升维
# # input_train_scaled_unsqueeze = input_train_scaled.unsqueeze(1)
# output_train_scaled = scaler.fit_transform(y_train)

# input_test_scaled = scaler.fit_transform(X_test)
# # 升维
# # input_test_scaled_unsqueeze = input_test_scaled.unsqueeze(1)
# output_test_scaled = scaler.fit_transform(y_test)

# 不缩放数据
input_train_scaled = X_train
output_train_scaled =y_train
input_test_scaled =X_test
output_test_scaled =y_test

# # 将numpy数组转成torch的张量
input_train_tensor = torch.from_numpy(input_train_scaled).float()
output_train_tensor = torch.from_numpy(output_train_scaled).float()
# # input_val_tensor = torch.from_numpy(input_val_scaled).float()
# # output_val_tensor = torch.from_numpy(output_val_scaled).float()
input_test_tensor = torch.from_numpy(input_test_scaled).float()
output_test_tensor = torch.from_numpy(output_test_scaled).float()

dataset = TensorDataset(input_train_tensor,output_train_tensor)
DataLoader = DataLoader(dataset, batch_size=64,shuffle=True)

# 初始化模型、优化器和损失函数
net = CNN1Dmo()

# net = GNN(input_dim=3, hidden_dim=16, output_dim=6)


# optimizer = torch.optim.Adam(net.parameters(), lr=0.01,)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adadelta(net.parameters(), lr=1.0)
# optimizer = torch.optim.shampoo(net.parameters())

# 构造RMSELoss

loss_func = nn.MSELoss()

# GPU训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
loss_func = loss_func.to(device)

# 画图准备
x = []                  # 横坐标
train_loss_list = []    # train损失值
val_loss_list = []      # val损失值
# train_acc_list = []     # train准确率


# GPU训练
input_train_tensor = input_train_tensor.to(device)
output_train_tensor = output_train_tensor.to(device)


# input_val_tensor = input_val_tensor.to(device)
# output_val_tensor = output_val_tensor.to(device)

max_epoch = 500
# batch_size = 5000
# numInterationsPerEpoch = len(input_train_tensor) // batch_size

for epoch in range(max_epoch):
    for input_batch, output_batch in DataLoader:
        input_batch, output_batch = input_batch.to(device), output_batch.to(device)
        optimizer.zero_grad()
        prediction = net(input_batch.unsqueeze(1))
        loss = loss_func(prediction.squeeze(1), output_batch)
        loss.backward()
        optimizer.step()
    # whole_data_index = list(range(len(input_train_tensor)))
    # random.shuffle(whole_data_index)
    
    # for batch in range(numInterationsPerEpoch):
    #     batch_index = whole_data_index[batch*batch_size:(batch+1)*batch_size]
    #     prediction = net(input_train_tensor[batch_index, :])    # 前向传播
    #     loss = loss_func(prediction, output_train_tensor[batch_index, :])   # 计算误差
    #     optimizer.zero_grad()   # 为下面的反向传播清除梯度（否则梯度会叠加）
    #     loss.backward()         # 反向传播计算梯度
    #     optimizer.step()        # 利用计算的梯度代入优化器，继而更新参数
        
        # val_prediction = net(input_val_tensor)
        # val_loss = loss_func(val_prediction, output_val_tensor)

    if epoch % 50 == 0:
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
        
    if epoch == max_epoch - 1:
        # 取出预测数据与label数据并化成numpy形式
        prediction = net(input_train_tensor.unsqueeze(1))    # 前向传播
        prediction = prediction.data.cpu().numpy()
        output_train_tensor = output_train_tensor.data.cpu().numpy()

        # # # 反归一化
        # prediction = scaler.inverse_transform(prediction)
        # output_train_tensor = scaler.inverse_transform(output_train_tensor)
        
        # 把预测数据与label数据化成torch的tenser形式
        prediction = torch.from_numpy(prediction).float()
        output_train_tensor = torch.from_numpy(output_train_tensor).float()


        loss_angle = loss_func(prediction[:, 0:3], output_train_tensor[:, 0:3])
        loss_phix = loss_func(prediction[:, 0], output_train_tensor[:, 0])
        loss_phiy = loss_func(prediction[:, 1], output_train_tensor[:, 1])
        loss_phiz = loss_func(prediction[:, 2], output_train_tensor[:, 2])
        print('训练集角度RMSE:')
        print(torch.sqrt(loss_angle))
        print('训练集角度phiz,phiy,phiz的RMSE:')
        print(torch.sqrt(loss_phix))
        print(torch.sqrt(loss_phiy))
        print(torch.sqrt(loss_phiz))
        # print('例子:')
        # print(prediction[0, 0:3], output_train_tensor[0, 0:3])

        loss_position = loss_func(prediction[:, 3:6], output_train_tensor[:, 3:6])
        loss_px = loss_func(prediction[:, 3], output_train_tensor[:, 3])
        loss_py = loss_func(prediction[:, 4], output_train_tensor[:, 4])
        loss_pz = loss_func(prediction[:, 5], output_train_tensor[:, 5])
        print('训练集位移RMSE:')
        print(torch.sqrt(loss_position))
        print('训练集位移px,py,pz的RMSE:')
        print(torch.sqrt(loss_px))
        print(torch.sqrt(loss_py))
        print(torch.sqrt(loss_pz))
        # print('例子:')
        # print(prediction[0, 3:6], output_train_tensor[0, 3:6])

plt.ioff()
plt.savefig('train_loss_cluster1_0319')
plt.show()
plt.close()

# plt.plot(x, val_loss_list, 'b', lw=1)
# plt.savefig('val_loss')
# plt.close()

# 把数据放入GPU中
input_test_tensor = input_test_tensor.to(device)

# 测试网络模型
test_prediction = net(input_test_tensor.unsqueeze(1))

test_prediction = test_prediction.data.cpu().numpy()

# # 反归一化
# test_prediction = scaler.inverse_transform(test_prediction)
# output_test_tensor = scaler.inverse_transform(output_test_tensor)

# 把预测数据与label数据化成torch的tensor形式
test_prediction = torch.from_numpy(test_prediction).float()
# output_test_tensor = torch.from_numpy(output_test_tensor).float()
output_test_tensor = torch.Tensor(output_test_tensor).float()

print("********************************************************************")
print("测试集角度预测RMSE:")
print(torch.sqrt(loss_func(test_prediction[:, 0:3], output_test_tensor[:, 0:3])).data.cpu())
print("测试集角度预测phix,phiy,phiz的RMSE:")
print(torch.sqrt(loss_func(test_prediction[:, 0], output_test_tensor[:, 0])).data.cpu())
print(torch.sqrt(loss_func(test_prediction[:, 1], output_test_tensor[:, 1])).data.cpu())
print(torch.sqrt(loss_func(test_prediction[:, 2], output_test_tensor[:, 2])).data.cpu())
print("测试集位移预测RMSE:")
print(torch.sqrt(loss_func(test_prediction[:, 3:6], output_test_tensor[:, 3:6])).data.cpu())
print("测试集位移预测px,py,pz的RMSE:")
print(torch.sqrt(loss_func(test_prediction[:, 3], output_test_tensor[:, 3])).data.cpu())
print(torch.sqrt(loss_func(test_prediction[:, 4], output_test_tensor[:, 4])).data.cpu())
print(torch.sqrt(loss_func(test_prediction[:, 5], output_test_tensor[:, 5])).data.cpu())

# 保存网络模型结构以及参数
torch.save(net, 'net_cluster1_0319.pkl')

# # 取出模型
# net_record = torch.load('net.pkl')
