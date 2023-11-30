import os
import pandas as pd
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np
import captum.attr as ca


import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from data.pre_data import NanotopeDataset
from model.GNNnet import Nanotope

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

def Training(model,epochs,batch_size,train_data,test_data,k):

    model.cuda()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  
    val_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False) 

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # loss_function = BCEFocalLoss(alpha=0.15)
    loss_function = nn.BCELoss(reduce = 'mean')
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
    train_losses,val_losses = [],[]

    flag =False
    pre_prauc = 0.0
    epoch = 0
    for e in range(epochs):
        model.train()
        train_loss,val_loss = 0.,0.
        for i, data in tqdm(enumerate(train_loader)):

            out = model(data)
            out = out.view(-1,140)
            out = torch.cat([out[i,:m] for i, m in enumerate(data.mask)])
            
            optimizer.zero_grad()
            loss = loss_function(out, data.y.float().cuda())
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss+=loss.item()

        scheduler.step()   
        with torch.no_grad():
            model.eval()
            cm = np.zeros((2,2))
            sum_pro = []
            sum_label = []
            for i,data in tqdm(enumerate(val_loader)):
                out = model(data)
                out = out.view(-1,140)
                out = torch.cat([out[i,:m] for i, m in enumerate(data.mask)])
                
                loss = loss_function(out, data.y.float().cuda())
                val_loss+=loss.item()
            
                sum_label.append(data.y.float().cpu())
                sum_pro.append(out.cpu())


        sum_prob = sum_pro.copy()
        sum_label = sum_label.copy()

        sum_pro = torch.concat(sum_pro)
        sum_label = torch.concat(sum_label)

        auc_roc = roc_auc_score(sum_label,sum_pro)
        pr_auc = average_precision_score(sum_label,sum_pro)

        print(pr_auc,pre_prauc,end=' ')
        print(pr_auc > pre_prauc)
        if pr_auc > pre_prauc and k>=0:
            print(1)
            folder_path = os.path.join(r'paratope\kfold',str(k))
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            torch.save(sum_label,os.path.join(folder_path,'label.pt'))
            torch.save(sum_prob,os.path.join(folder_path,'pre.pt'))
            pre_prauc = pr_auc
            epoch = e
        # else:
        #     pre_prauc = pr_auc
        #     epoch = e

        dict_path = r'E:\608\paratope\model\pt'
        torch.save(model.state_dict(), os.path.join(dict_path,str(e)+'.pt'))

        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))

        print(  "训练集学习次数: {}/{}.. ".format(e+1, epochs),
                "训练误差: {:.5f}.. ".format(train_loss/len(train_loader)),
                "测试集误差: {:.5f}.. ".format(val_loss/len(val_loader)),
                '\t',
                "测试集auc-roc值:{:.5f}..".format(auc_roc),
                "测试集pr auc值:{:.5f}..".format(pr_auc),
                )

        if flag:
            return train_losses,val_losses
    print(pre_prauc)
    print('最好的结果出现在第'+str(epoch)+',结果为:',pre_prauc)

    return train_losses,val_losses

def plot(train_losses,test_losses):
    import matplotlib.pyplot as plt
    
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend()
    plt.savefig('loss.jpg')

def kfold(nano_dataset,fv_dataset,k,batch_size):
    size1 = len(fv_dataset)//k
    size2 = len(nano_dataset)//k
    
    print('开始训练')
    for fold in range(k):
        
        val_start1 = fold*size1
        val_end1 = val_start1+size1

        val_start2 = fold*size2
        val_end2 = val_start2+size2

        print(f'第{fold}次训练')

        train_data = ConcatDataset([
            Subset(fv_dataset,range(0,val_start1)),
            Subset(fv_dataset,range(val_end1,len(fv_dataset))),

            Subset(nano_dataset,range(0,val_start2)),
            Subset(nano_dataset,range(val_end2,len(nano_dataset))),
            ])

        val_data = Subset(nano_dataset,range(val_start2,val_end2))

        model = Nanotope(hidden_channels=512, num_layers=3, num_heads=8,num_bases=8)
        Training(model,10,batch_size,train_data,val_data,fold)


if __name__ =='__main__':
    batch_size = 64
    k = 32
    flag = True


    Nano_dataset = NanotopeDataset(r'data\edge'+str(k)+'\\Nano')
    heavy_dataset = NanotopeDataset(r'data\edge'+str(k)+'\\Heavy_Fv')
    #分割数据集
    subNano_dataset = Subset(Nano_dataset, range(400))
    subheavy_dataset = Subset(heavy_dataset,range(600))

    train_data = ConcatDataset([subheavy_dataset, subNano_dataset])
    test_data = Subset(Nano_dataset, range(400,len(Nano_dataset)))


    model = Nanotope(hidden_channels=512, num_layers=3, num_heads=8,num_bases=8)
    if flag:
        print(len(test_data))
        print("开始训练")
        train_losses,test_losses= Training(model,10,batch_size,train_data,test_data,k)
        print('k的个数为:',k)
        # kfold(subNano_dataset,subheavy_dataset,10,10)

    else:
        print('测试')
        model.load_state_dict(torch.load(r'\model\pt\3.pt'))
        model.eval().cuda()
        train_loader = DataLoader(test_data, batch_size=1, shuffle=True)  
        
        for i,data in enumerate(train_loader):
            break
        out,attention = model(data)

        
        
        
