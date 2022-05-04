import os
import glob
import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

# start = time.time()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

from models.conv1 import CNNNet

label_num = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print('——————————————我是分割线——————————————')


net = CNNNet()
net.to(device)
print(f"device:{device}")

#############处理测井数据
print(os.getcwd())     #返回当前工作目录

# 定义了读取txt，按逗号划分并返回为数组
def read_txt(file):
    f = open(file)
    result = []
    for line in f.readlines():
        line = line.strip().split(',')              
        result.append(line)
    return np.array(result)



Well_all = glob.glob('/home/yechen/code/cejing/cejing/data/Well_228_old/Well_228_old/train/*.txt')  # 读了train的所有井（一部分train,一部分valid验证）
Well_test = glob.glob('/home/yechen/code/cejing/cejing/data/Well_228_old/Well_228_old/test/*.txt')
# Well_all.sort()
# print(Well_all)

name = {'K1z2+1': 0, 
        'J2a': 1, 
        'J2z': 2, 
        'J1y': 3, 
        'J1f': 4, 
        'chang1': 5, 
        'chang2': 6, 
        'chang3': 7, 
        'chang4+5': 8,
        'chang6': 9}

well_used = []
ori_label = []
ori_x = []
time_step = 96
well_i = 0

for i in range(0, len(Well_all)):
    if name == {}:
        break
    Well = read_txt(Well_all[i])
    Well_x, Well_y = [], []
    for line in Well:
        Well_x.append(np.array([float(x) for x in line[2:-1]]))  # x是训练数据（第3列到倒数第二列）
        Well_y.append(line[-1])  # y是标签（最后一列）

    # print(well_i, Well_all[i])
    well_i += 1
    well_used.append(Well_all[i][18:-4]) ## ???

    Well_x = np.array(Well_x)
    Well_x = scale(Well_x, axis=0, with_mean=True, with_std=True, copy=True)  # 按井标准化

    for i in range(0, len(Well_x) - (time_step - 1), 32):
        if Well_y[i] not in name:
            name[Well_y[i]] = label_num
            label_num += 1
            # continue
        # if ori_label.count(name[Well_y[i]]) < 50000:
        ori_x.append([x for x in Well_x[i:i + time_step]])
        labels = [name[Well_y[index]] for index in range(i, i + time_step)]
        counts = np.bincount(labels)    #找出最多的类别，给出每个索引值出现的次数
        ori_label.append(np.argmax(counts))    #找出出现最多的类别数（标签取众数）
        # ori_label.append(name[Well_y[i]])
        # ori_label.append(max(name[Well_y[i]].count(x) for x in set(name[Well_y[i]])))
        # print('看看是取了众数了吗', ori_label)

        # else:
        #  del name[Well_y[i]]

    # for key in name.keys():
    # print(key + '(' + str(name[key]) + ')' + ':', ori_label.count(name[key]))

ori_x = np.array(ori_x)
ori_label = np.array(ori_label, dtype=np.float32)
# print(ori_x[:10])
print("ori_x.shape:", ori_x.shape)
print("ori_label:", ori_label.shape)
ori_x = ori_x[:, np.newaxis, :, :]  ###########看这里！！！
# ori_label = ori_label[:,np.newaxis]
# label = to_categorical(ori_label)

#print("well_used:", well_used)

# np.savez("npz/train_data_w200.npz", x=ori_x, y=label)

'''
train_data = np.load("npz/train_data_50000.npz")
ori_x = train_data['x']
label = train_data['y']
'''

trainX, validX, trainY, validY = train_test_split(ori_x, ori_label, test_size=0.1, random_state=42)
print(validY)
print(trainX.shape)
print(validX.shape)
print(trainY.shape)
print(validY.shape)

trainX = torch.from_numpy(trainX).float()
trainY = torch.from_numpy(trainY).float()
validX = torch.from_numpy(validX).float()
validY = torch.from_numpy(validY).float()
# testX = torch.from_numpy(testX).float()
# testY = torch.from_numpy(testY).float()


train_set = TensorDataset(trainX, trainY)
valid_set = TensorDataset(validX, validY)

# test_set = TensorDataset(testX, testY)

#########构建Dataloader
trainloader = DataLoader(dataset=train_set, batch_size=4096, shuffle=True,
                         num_workers=0)  ######!!!!!这里不能只放进数据而没有标签吧！！！！！
validloader = DataLoader(dataset=valid_set, batch_size=4096, shuffle=False, num_workers=0)
# testloader = DataLoader(dataset=test_set, batch_size=2048, shuffle=False, num_workers=0)

##########定义损失和优化器
loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=0.001)

###########训练网络
accs = []
loss_count = []
net.train()

for epoch in range(7):
    since = time.time()
    print('epoch ', epoch + 1)
    torch.save(net.state_dict, f'/home/yechen/code/cejing/cejing/data/Well_228_old/Well_228_old/result/{epoch}.pth')
    for i, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)

        net.zero_grad()
        opt.zero_grad()  # 清空上一步残余更新参数值

        out = net(x)
        # 获取损失
        loss = loss_func(out, y.long())
        # 使用优化器优化损失

        loss.backward()  # 误差反向传播，计算参数更新值
        opt.step()  # 将参数更新值施加到net的parmeters上
        # print(loss.item())

        if i % 10 == 0:
            loss_count.append(loss)
            # print('{}:\t'.format(i), loss.item())
            # torch.save(net, r'E:\try\resultsss\train_net_result')
            net.eval()
            for a, b in validloader:
                a = a.to(device)
                with torch.no_grad():
                    out = net(a)
                print('test_out:\t', torch.max(out, 1)[1])
                print('test_y:\t', b)
                out = out.to("cpu")
                accuracy = torch.max(out, 1)[1].numpy() == b.numpy()
                print('accuracy:\t', accuracy.mean())
                accs.append(accuracy.mean())
                break
            net.train()

time_elapsed = time.time() - since
print('every epoch training time: {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

plt.figure('PyTorch_CNN_Loss')
plt.plot(loss_count, label='Loss')
plt.legend()
plt.savefig('PyTorch_CNN_Loss.png')
plt.show()

plt.figure('PyTorch_CNN_Acc')
plt.plot(accs, label='Acc')
plt.legend()
plt.savefig('PyTorch_CNN_Acc.png')
plt.show()
# 出现保存的图片是空白原因：
# 在 plt.show() 后调用了 plt.savefig() ，
# 在 plt.show() 后实际上已经创建了一个新的空白的图片（坐标轴），
# 这时候你再 plt.savefig() 就会保存这个新生成的空白图片。
# 两个解决办法，1、先savefig再show
# 2、画图的时候获取当前图像：
# gcf: Get Current Figure
# fig = plt.gcf()
# plt.show()
# fig1.savefig('tessstttyyy.png', dpi=100)



with torch.no_grad():
    net.eval()
    net.to('cpu')
    for j in range(0, len(Well_test)):
        Well = read_txt(Well_test[j])
        Well_x, Well_y = [], []
        X, Y = [], []
        for line in Well:
            Well_x.append(np.array([float(x) for x in line[2:-1]]))  # x是训练数据（第3列到倒数第二列）
            Well_y.append(line[-1])  # y是标签（最后一列）

        Well_x = np.array(Well_x)
        Well_x = scale(Well_x, axis=0, with_mean=True, with_std=True, copy=True)  # 按井标准化

        for i in range(0, len(Well_x) - (time_step - 1), 32):
            if Well_y[i] not in name:
                name[Well_y[i]] = label_num
                label_num += 1
                # continue
            # if ori_label.count(name[Well_y[i]]) < 50000:
            X.append([x for x in Well_x[i:i + time_step]])
            labels = [name[Well_y[index]] for index in range(i, i + time_step)]
            counts = np.bincount(labels)
            Y.append(np.argmax(counts))
        X = np.array(X)
        Y = np.array(Y, dtype=np.float32)
        X = X[:, np.newaxis, :, :]
        testX = torch.from_numpy(X).float()
        accuracy = torch.max(net(testX), 1)[1].numpy() == Y
        print(Well_test[j].split('/')[-1] + ',' + str(accuracy.mean()))