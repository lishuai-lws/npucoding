import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
data_path='D:/data/output/emo-db_functional'
loss_list=[]
#load data from data_path to x_data and y_data
def CSVDataopen(data_path):
    functional_path = data_path
    data_list=os.listdir(functional_path)
    x_data=np.empty([1,6902],dtype='float32')#save functional data
    y_data=np.empty([1,1],dtype='int64')
    for data in data_list:
        if data[-4:]=='.csv':
            now_path=os.path.join(functional_path,data)
            now_data = np.loadtxt(now_path, delimiter=",", skiprows=1, dtype='float32', unpack=True)
            now_data = now_data.reshape(1, 6903)
            now_data = now_data[:, 1:]
            x_data = np.concatenate((x_data, now_data), axis=0)
            y_data= np.append(y_data,stof(data[-6]))
    x_data=np.delete(x_data,0,axis=0)
    y_data=y_data.reshape(1,len(data_list)+1)
    y_data = np.delete(y_data, 0, axis=1)
    return x_data,y_data


def stof(s):
    S=['W','L','E','A','F','T','N']
    for i in range(0,7):
        if S[i]==s:
            return i



class CSVDataset(Dataset):
    def __init__(self,filepath):
        x_data,y_data=CSVDataopen(filepath)
        self.len=x_data.shape[0]
        self.x_data=torch.from_numpy(x_data)
        self.y_data=torch.from_numpy(y_data)


    def __getitem__(self, item):
        return self.x_data[item],self.y_data[0,item]


    def __len__(self):
        return self.len

dataset=CSVDataset(data_path)
train_loader=DataLoader(dataset=dataset,
                        batch_size=50,
                        shuffle=True,
                        num_workers=2
)



# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.linear = torch.nn.Linear(8, 1)
#         self.sifmoid=torch.nn.Sigmoid()
#
#     def forward(self, x):
#         y_pred = self.sigmoid(self.linear(x))
#         return y_pred


# model = Model()
model=torch.nn.Sequential(
    torch.nn.Linear(6902,512),
    torch.nn.Sigmoid(),
    torch.nn.Linear(512,256),
    torch.nn.Sigmoid(),
    torch.nn.Linear(256,128),
    torch.nn.Sigmoid(),
    torch.nn.Linear(128,64),
    torch.nn.Sigmoid(),
    torch.nn.Linear(64,7)
)
def train(epoch):
    running_loss=0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print('epoch:', epoch,  'Loss:', running_loss/10)
    loss_list.append(running_loss/10)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
if __name__=='__main__':
    for epoch in range(10):
        train(epoch)
    plt.plot(loss_list)
    plt.show()

