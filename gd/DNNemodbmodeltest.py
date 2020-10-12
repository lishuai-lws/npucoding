import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from sklearn import metrics
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import preprocessing





data_path='../emodb/emo-db_functional'
loss_list = []
batch_size = 50
learning_rate = 0.001
num_epochs = 50
foldnum=1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#load data from data_path to x_data and y_data
def CSVDataopen(data_path):
    functional_path = data_path
    data_list = os.listdir(functional_path)
    x_data = np.empty([1,6902],dtype='float32')#save functional data
    y_data = np.empty([1,1],dtype='int64')
    for data in data_list:
        if data[-4:]=='.csv':
            now_path = os.path.join(functional_path,data)
            now_data = np.loadtxt(now_path, delimiter=",", skiprows=1, dtype='float32', unpack=True)
            now_data = now_data.reshape(1, 6903)
            now_data = now_data[:, 1:]
            x_data = np.concatenate((x_data, now_data), axis=0)
            y_data = np.append(y_data,stof(data[-6]))
    x_data = np.delete(x_data,0,axis=0)
    y_data = y_data.reshape(1,len(data_list)+1)
    y_data = np.delete(y_data, 0)
    x_data = preprocessing.scale(x_data)#标准化
    return x_data,y_data
def stof(s):
    S=['W','L','E','A','F','T','N']
    for i in range(0,7):
        if S[i]==s:
            return i
class CSVDataset(Dataset):
    def __init__(self,x_data,y_data):
        self.len = x_data.shape[0]
        self.x_data = torch.from_numpy(x_data)
        self.y_data = torch.from_numpy(y_data)
    def __getitem__(self, item):
        return self.x_data[item],self.y_data[item]
    def __len__(self):
        return self.len

model=torch.load('DNNModel_EMODB.pkl')
def t_data(x_data, y_data):
    dataset = CSVDataset(x_data, y_data)
    x_data,y_data=dataset.x_data,dataset.y_data
    with torch.no_grad():
        x_data = x_data.to(device)
        y_data = y_data.to(device)
        output = model(x_data)
        output = torch.max(output.data,dim=1)[1]
        print(y_data)
        print(output)
        Total=len(output)
        ACC=metrics.accuracy_score(output,y_data)*100
        MAP=metrics.precision_score(output,y_data,average='macro')*100
        MAR=metrics.recall_score(output,y_data,average='macro')*100
        MAF=metrics.f1_score(output,y_data,average='macro')*100
        print('Total: %d,ACC: %.5f %%,MAP= %.5f %%,MAR=%.5f %%,MAF= %.5f %%' % (Total,ACC,MAP,MAR,MAF))
    return ACC,MAP,MAR,MAF

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate , momentum=0.5)
if __name__=='__main__':
    aveacc = 0
    avemap = 0
    avemar = 0
    avemaf = 0
    x_data , y_data = CSVDataopen(data_path)
    randomindex=np.random.permutation(x_data.shape[0])
    x_data=x_data[randomindex]
    y_data = y_data[randomindex]
    ACC, MAP, MAR, MAF=t_data(x_data[:100],y_data[:100])
    aveacc+=ACC
    avemap+=MAP
    avemaf+=MAF
    avemar+=MAR
    # plt.plot(loss_list)
    # plt.show()
    print('AVERAGE:ACC: %f %%,MAP: %f %%,MAR: %f %%,MAF: %f %%' % (aveacc/foldnum,avemap/foldnum,avemar/foldnum,avemaf/foldnum))
    # torch.save(model, './DNNModel_EMODB.pkl')
    print('Finish!')
