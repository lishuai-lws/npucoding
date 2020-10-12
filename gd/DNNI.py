import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold





data_path='D:/data/output/iemocap_functional'
loss_list=[]
batch_size=50
#load data from data_path to x_data and y_data
def CSVDataopen(data_path):
    functional_path = data_path
    file_list=os.listdir(functional_path)
    x_data=np.empty([1,6902],dtype='float32')#save functional data
    y_data=np.empty([1,1],dtype='int64')
    for file in file_list:
        audio_path = os.path.join(functional_path, file)
        data_list = os.listdir(audio_path)
        labels=label(file)
        for data in data_list:
            if data[-4:]=='.csv':
                now_path=os.path.join(audio_path,data)
                now_data = np.loadtxt(now_path, delimiter=",", skiprows=1, dtype='float32', unpack=True)
                now_data = now_data.reshape(1, 6903)
                now_data = now_data[:, 1:]
                x_data = np.concatenate((x_data, now_data), axis=0)
                y_data= np.append(y_data,labels)
    x_data=np.delete(x_data,0,axis=0)
    mean=np.mean(x_data,axis=0)
    std=np.std(x_data,axis=0)
    y_data=y_data.reshape(1,len(x_data)+1)
    y_data = np.delete(y_data, 0)
    for i in range(len(x_data)):
        if std[i]!=0:
            x_data[i] = (x_data[i] - mean[i]) / std[i]
    return x_data,y_data
def label(file):
    if file=='angry':
        return 0
    elif file=='happy':
        return 1
    elif file=='neutral':
        return 2
    else:
        return 3
class CSVDataset(Dataset):
    def __init__(self,x_data,y_data):
        self.len=x_data.shape[0]
        self.x_data=torch.from_numpy(x_data)
        self.y_data=torch.from_numpy(y_data)
    def __getitem__(self, item):
        return self.x_data[item],self.y_data[item]
    def __len__(self):
        return self.len
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(6902, 4096)
        self.linear2 = torch.nn.Linear(4096, 2048)
        self.linear3 = torch.nn.Linear(2048, 1024)
        self.linear4 = torch.nn.Linear(1024, 512)
        self.linear5 = torch.nn.Linear(512, 256)
        self.linear6 = torch.nn.Linear(256, 64)
        self.linear7 = torch.nn.Linear(64, 4)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        x = torch.sigmoid(self.linear5(x))
        x = torch.sigmoid(self.linear6(x))
        return self.linear7(x)

#
#
# model=torch.nn.Sequential(
#     torch.nn.Linear(6902,4096),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(4096, 2048),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(2048, 512),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(512,256),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(256,128),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(128,64),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(64,7),
# )
def train(epoch,x_data,y_data):
    dataset = CSVDataset(x_data, y_data)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2
                              )
    for epoch in range(epoch):
        batch_num=0
        running_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            batch_num=batch_num+1
        print('epoch:', epoch,  'Loss:', running_loss/batch_num)
        loss_list.append(running_loss/batch_num)
def t_data(x_data, y_data):
    total=0
    current=0
    dataset = CSVDataset(x_data, y_data)
    test_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=2
                             )
    with torch.no_grad():
        for x_data,y_data in test_loader:
            output=model(x_data)
            # print(output.data)
            output=torch.max(output.data,dim=1)[1]
            print(output)
            total+=y_data.size(0)
            current+=np.sum(output.numpy()==y_data.numpy())
    print('acc: %d %%' % (100*current/total))


model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
if __name__=='__main__':
    epoch=10
    x_data, y_data = CSVDataopen(data_path)
    train(epoch,x_data,y_data)
    t_data(x_data[:50],y_data[:50])
    plt.plot(loss_list)
    plt.show()

