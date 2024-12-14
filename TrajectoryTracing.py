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
import csv
import xlwt


from CNN1DNetworkMO import CNN1Dmo
from distance import RpToTrans, TransToRp, TransInv, se3ToVec

# 载入轨迹
data = loadmat('./TrajectorySin_0226.mat')
dataset = data['TrajectorySin_0226']



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_data = torch.from_numpy(dataset[:, 0:3]).float()
output_data = torch.from_numpy(dataset[:, 3:9]).float()

dataset = TensorDataset(input_data, output_data)
data_loader = DataLoader(dataset)

input_data = input_data.to(device)
# output_data = output_data.to(device)

# 初始化八个模型
num_nets = 8
nets =[]




MSEfunc = nn.MSELoss()



# 加载训练好的权重
for i in range(num_nets):
    net_path = f'./log/log_0117/net/net_cluster{i+1}_0117.pkl'  
    net = torch.load(net_path, map_location=device)
    net.eval()
    nets.append(net)


# 将三个轨迹的参数同时输入到八个网络中
predictions = []

distances = []

errors = []


first_iteration = True

for input_batch, output_batch in data_loader:
    input_batch = input_batch.to(device)
    min_distance = float('inf')

    # plt.figure(figsize=(10, 6))
    # plt.plot(output_batch[:,0].cpu().numpy(), label='out_batch_px', linestyle='-', color='black')

    # previous_prediction = output_batch
    if first_iteration:
        previous_prediction = output_batch.clone()
        selected_prediction = output_batch.clone()
        Tlast = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        first_iteration = False
    else:
        previous_prediction = selected_prediction.clone()

    for net in nets:
        with torch.no_grad():
            prediction = net(input_batch.unsqueeze(1))

            prediction = prediction.data.cpu().numpy()

            prediction = torch.from_numpy(prediction).float()
            
            # plt.plot(prediction[:,5], label='out_batch_pz', linestyle='-', color='red')

            T = RpToTrans(prediction[:, 0], prediction[:, 1], prediction[:, 2],
                            prediction[:, 3], prediction[:, 4], prediction[:, 5])
            
            TlastInv = TransInv(Tlast)
            

            distance1 = np.linalg.norm(se3ToVec(T@TlastInv))
            
            distance2 = torch.sqrt(MSEfunc(prediction[:, 0:3], output_batch[:, 0:3])).data.cpu().numpy() +  \
                        torch.sqrt(MSEfunc(prediction[:, 3:6], output_batch[:, 3:6])).data.cpu().numpy()
                        
            distance3 = torch.sqrt(MSEfunc(prediction[:, 5], output_batch[:,5])).data.cpu()

            distance =  distance2
                          
            if distance < min_distance:
                min_distance = distance
                selected_prediction = prediction.clone()
                Tlast = T
    # plt.legend()
    # plt.title('Comparison of out_batch and Predictions')
    # plt.xlabel('Data Points')
    # plt.ylabel('Values')
    # plt.show()
    predictions.append(selected_prediction)
    error = torch.abs((selected_prediction - output_batch) / output_batch)

    errors.append(error)

   
    distances.append(min_distance)


predictions = torch.cat(predictions,dim=0)
errors = torch.cat(errors,dim=0)

# errors_list = errors.tolist()
# predictions_list = predictions.tolist()
# result = open('data_p.xls', 'w', encoding='gbk')
# # result.write('X\tY\n')
# for m in range(len(predictions_list)):
#     for n in range(len(predictions_list[m])):
#         result.write(str(predictions_list[m][n]))
#         result.write('\t')
#     result.write('\n')
# result.close()

# print(errors[0,0].tolist())
# print(predictions[0,0].tolist())

# phix_array = 100 * errors[:,0] / torch.abs(predictions[:,0])
# count_phix = 100 * (torch.sum(phix_array<1.0).item()) / len(predictions[:,0])

# phiy_array = 100 * errors[:,1] / torch.abs(predictions[:,1])
# count_phiy = 100 * (torch.sum(phiy_array<1.0).item()) / len(predictions[:,1])

# phiz_array = 100 * errors[:,2] / torch.abs(predictions[:,2])
# count_phiz = 100 * (torch.sum(phiz_array<1.0).item()) / len(predictions[:,2])

# count_phi = (100 * (torch.sum(phix_array<1.0).item() + torch.sum(phiy_array<1.0).item() + torch.sum(phiz_array<1.0).item()) )/ (3 * len(predictions[:,2]))

# px_array = 100 * errors[:,3] / torch.abs(predictions[:,3])
# count_px = 100 * (torch.sum(px_array<1.0).item()) / len(predictions[:,3])

# py_array = 100 * errors[:,4] / torch.abs(predictions[:,4])
# count_py = 100 * (torch.sum(py_array<1.0).item()) / len(predictions[:,4])

# pz_array = 100 * errors[:,5] / torch.abs(predictions[:,5])
# count_pz = 100 * (torch.sum(pz_array<1.0).item()) / len(predictions[:,5])

# count_p = (100 * (torch.sum(px_array<1.0).item() + torch.sum(py_array<1.0).item() + torch.sum(pz_array<1.0).item())) /( 3 * len(predictions[:,2]))

# print("路径预测过程中角度和位移预测误差小于1.0%的分别有")
# print(round(count_phix,2),"%")
# print(round(count_phiy,2),"%")
# print(round(count_phiz,2),"%")
# print(round(count_px,2),"%")
# print(round(count_py,2),"%")
# print(round(count_pz,2),"%")
# print(round(count_phi,2),"%")
# print(round(count_p,2),"%")




# 绘制 output_data[:, 5] 和 predictions[:, 5]
# plt.figure(figsize=(10, 6))

# # 绘制 output_data[:, 5]
# plt.plot(output_data[:, 5].cpu().numpy(), label='Ground Truth', marker='o')

# # 绘制 predictions[:, 6]
# plt.plot(predictions[:, 5].cpu().numpy(), label='Predictions', marker='x')

# plt.title('Output and Predictions for pz')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.legend()
# plt.savefig('TrajectoryTracingpz_0227')
# plt.show()
# plt.close()




# 画出轨迹图
#---------------------------

# plt.figure(figsize=(10, 15))

# plt.subplot(311)
# plt.plot(output_data[:, 0], label='phix', marker='.')
# plt.title('Trajectory for phix')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.legend()

# plt.subplot(312)
# plt.plot(output_data[:, 1], label='phiy', marker='.')
# plt.title('Trajectory for phiy')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.legend()

# plt.subplot(313)
# plt.plot(output_data[:, 5], label='pz', marker='.')
# plt.title('Trajectory for pz')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.legend()

# plt.savefig('./Trajectory_0226')
# plt.show()
# plt.close()

#---------------------------


# plt.figure(figsize=(10, 15))

# plt.subplot(311)
# plt.plot(output_data[:, 0], label='Ground Truth', marker='.')
# plt.plot(predictions[:, 0], label='Predictions', marker='.')
# plt.title('Output and Predictions for phix')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.legend()

# plt.subplot(312)
# plt.plot(output_data[:, 1], label='Ground Truth', marker='.')
# plt.plot(predictions[:, 1], label='Predictions', marker='.')
# plt.title('Output and Predictions for phiy')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.legend()

# plt.subplot(313)
# plt.plot(output_data[:, 2], label='Ground Truth', marker='.')
# plt.plot(predictions[:, 2], label='Predictions', marker='.')
# plt.title('Output and Predictions for phiz')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.legend()

# plt.savefig('./TrajectoryTracing_phi_0301_1')
# plt.show()
# plt.close()

# plt.figure(figsize=(10, 15))

# plt.subplot(311)
# plt.plot(output_data[:, 3], label='Ground Truth', marker='.')
# plt.plot(predictions[:, 3], label='Predictions', marker='.')
# plt.title('Output and Predictions for px')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.legend()

# plt.subplot(312)
# plt.plot(output_data[:, 4], label='Ground Truth', marker='.')
# plt.plot(predictions[:, 4], label='Predictions', marker='.')
# plt.title('Output and Predictions for py')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.legend()

# plt.subplot(313)
# plt.plot(output_data[:, 5], label='Ground Truth', marker='.')
# plt.plot(predictions[:, 5], label='Predictions', marker='.')
# plt.title('Output and Predictions for pz')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.legend()

# plt.savefig('./TrajectoryTracing_p_0301_1')
# plt.show()
# plt.close()




# print("轨迹角度预测RMSE:")
# print(torch.sqrt(MSEfunc(predictions[:, 0:3], output_data[:, 0:3])).data.cpu())
# print("轨迹角度预测MAPE:")
# print()
# print(torch.mean((errors[:,0:3])*100,dim=0).data.cpu())

# print(torch.argmax((errors[:,3])*100).data.cpu())
# # print(torch.mean(torch.abs((predictions[:, 0:3] - output_data[:, 0:3]) / output_data[:, 0:3]) * 100).data.cpu())


# print("轨迹位移预测RMSE:")
# print(torch.sqrt(MSEfunc(predictions[:, 3:6], output_data[:, 3:6])).data.cpu())
# print("轨迹位移预测MAPE:")
# # print(torch.mean(torch.abs((predictions[:, 3:6] - output_data[:, 3:6]) / output_data[:, 3:6]) * 100).data.cpu())
# print(torch.mean((errors[:,3:6])*100,dim=0).data.cpu())

# print("轨迹phix预测RMSE:")
# print(torch.sqrt(MSEfunc(predictions[:, 0], output_data[:, 0])).data.cpu())
# print("轨迹phix预测MAPE:")
# print(torch.mean(torch.abs((predictions[:, 0] - output_data[:, 0]) / output_data[:, 0]) * 100).data.cpu())


# print("轨迹phiy预测RMSE:")
# print(torch.sqrt(MSEfunc(predictions[:, 1], output_data[:, 1])).data.cpu())
# print("轨迹phiy预测MAPE:")
# print(torch.mean(torch.abs((predictions[:, 1] - output_data[:, 1]) / output_data[:, 1]) * 100).data.cpu())

# print("轨迹phiz预测RMSE:")
# print(torch.sqrt(MSEfunc(predictions[:, 2], output_data[:, 2])).data.cpu())
# print("轨迹phiz预测MAPE:")
# print(torch.mean(torch.abs((predictions[:, 2] - output_data[:, 2]) / output_data[:, 2]) * 100).data.cpu())

# print("轨迹px预测RMSE:")
# print(torch.sqrt(MSEfunc(predictions[:, 3], output_data[:, 3])).data.cpu())
# print("轨迹px预测MAPE:")
# print(torch.mean(torch.abs((predictions[:, 3] - output_data[:, 3]) / output_data[:, 3]) * 100).data.cpu())


# print("轨迹py预测RMSE:")
# print(torch.sqrt(MSEfunc(predictions[:, 4], output_data[:, 4])).data.cpu())
# print("轨迹py预测MAPE:")
# print(torch.mean(torch.abs((predictions[:, 4] - output_data[:, 4]) / output_data[:, 4]) * 100).data.cpu())


# print("轨迹pz预测RMSE:")
# print(torch.sqrt(MSEfunc(predictions[:, 5], output_data[:, 5])).data.cpu())
# print("轨迹pz预测MAPE:")
# print(torch.mean(torch.abs((predictions[:, 5] - output_data[:, 5]) / output_data[:, 5]) * 100).data.cpu())




