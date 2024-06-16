import numpy as np
import torch
import torch.nn as nn
import argparse
from util import load_data_n_model
from util import set_seed
import torch.nn.functional as fun

# 'S' level refers to subcarrier level
# 'A' level refers to antenna level
# 'I' level refers to instance level
# 'T' level refers to timestamp level

def mixup_data(x, y, alpha=0.5, use_cuda=False, sub_carrier=114, level='I'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    if level == 'S':
        mixed_x = x
        sub_num = torch.tensor(int(sub_carrier * lam))
        mixed_x[:,:,:sub_num,:] = x[:,:,:sub_num,:]
        mixed_x[:,:,sub_num:,:] = x[index,:,sub_num:,:]
        # mixed_x = lam * x + (1 - lam) * x[index, :]
    elif level == 'A':
        mixed_x = x
        attennas = x.size()[1]
        att_num = torch.tensor(int(attennas * lam))
        mixed_x[:,:att_num,:,:] = x[:,:att_num,:,:]
        mixed_x[:,att_num:,:,:] = x[index,att_num:,:,:]

    elif level == 'T':
        mixed_x = x
        time_len = x.size()[3]
        time_num = torch.tensor(int(time_len * lam))
        mixed_x[:,:,:,:time_num] = x[:,:,:,:time_num]
        mixed_x[:,:,:,time_num:] = x[index,:,:,time_num:]

    elif level == 'I':
        mixed_x = x
        time_len = x.size()[3]
        time_num = torch.tensor(int(time_len * lam))
        mixed_x[:, :, :, :] = lam * x[:, :, :, :] + (1 - lam)*x[index, :, :, :]

        # mixed_x[:,:,:sub_num,:] = x[:,:,:sub_num,:]
        # mixed_x[:,:,sub_num:,:] = x[index,:,sub_num:,:]

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# def train(model, tensor_loader, num_epochs, learning_rate, criterion, device, level):
#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
#     for epoch in range(num_epochs):
#         model.train()
#         epoch_loss = 0
#         epoch_accuracy = 0
#         for data in tensor_loader:
#             inputs,labels = data
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             labels = labels.type(torch.LongTensor)
#             inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, level=level)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             outputs = outputs.to(device)
#             outputs = outputs.type(torch.FloatTensor)
#             # outputs = fun.one_hot(outputs,num_classes=6)
#             loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.item() * inputs.size(0)
#             predict_y = torch.argmax(outputs,dim=1).to(device)
#             epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
#         epoch_loss = epoch_loss/len(tensor_loader.dataset)
#         epoch_accuracy = epoch_accuracy/len(tensor_loader)
#        # print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))
#     return
#
#
# def test(model, tensor_loader, criterion, device, report):
#     model.eval()
#     test_acc = 0
#     test_loss = 0
#     file = open('report.txt','a')
#     for data in tensor_loader:
#         inputs, labels = data
#         inputs = inputs.to(device)
#         labels.to(device)
#         labels = labels.type(torch.LongTensor)
#         outputs = model(inputs)
#         outputs = outputs.type(torch.FloatTensor)
#         outputs.to(device)
#
#         loss = criterion(outputs,labels)
#         predict_y = torch.argmax(outputs,dim=1).to(device)
#         accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
#         test_acc += accuracy
#         test_loss += loss.item() * inputs.size(0)
#     test_acc = test_acc/len(tensor_loader)
#     test_loss = test_loss/len(tensor_loader.dataset)
#     print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))
#     file.write(str(round(test_acc,4))+'\t')
#     file.close()
#     return
