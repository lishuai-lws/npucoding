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
import itertools
from sklearn import preprocessing

data_path='../emodb/emo-db_functional'
loss_list = []
batch_size = 18
num_epochs = 900
learning_rate = 0.001
Dropoutrate=0.5
foldnum=10
is_trainsettest=False
ACClist,MAPlist,MARlist,MAFlist=[],[],[],[]
classes=['angry','bored','neutral','happy','fearful','sad','disgusted']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#load data from data_path to x_data and y_data
def CSVDataopen(data_path):
    functional_path = data_path
    data_list = os.listdir(functional_path)
    data_list.sort()
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
    S=['W','L','N','F','A','T','E']
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
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear=torch.nn.Sequential(
                torch.nn.Linear(6902, 1024),
                torch.nn.Dropout(Dropoutrate),
                torch.nn.BatchNorm1d(1024),
                torch.nn.Sigmoid(),
                torch.nn.Linear(1024, 64),
                torch.nn.Dropout(Dropoutrate),
                torch.nn.BatchNorm1d(64),
                torch.nn.Sigmoid(),
                # torch.nn.Linear(256, 64),
                # torch.nn.Dropout(Dropoutrate),
                # torch.nn.BatchNorm1d(64),
                # torch.nn.Sigmoid(),
                torch.nn.Linear(64, 7),
        )
    def forward(self, x):
        x=self.linear(x)
        return x
def train(model,num_epochs,x_data,y_data):
    train_dataset = CSVDataset(x_data, y_data)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2
                              )
    criterion = torch.nn.CrossEntropyLoss()
    criterion=criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        batch_num = 0
        running_loss = 0
        for i, data in enumerate(train_loader):
            x_data,y_data = data
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            output = model(x_data)
            loss = criterion(output, y_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            batch_num=batch_num+1
        print('Epoch [%d/%d],Loss: %.4f' % (epoch + 1,num_epochs, running_loss/(len(train_dataset) // batch_size + 1)))
        loss_list.append(running_loss)
def t_data(model,x_data, y_data):
    dataset = CSVDataset(x_data, y_data)
    x_data,y_data=dataset.x_data,dataset.y_data
    with torch.no_grad():
        x_data = x_data.to(device)
        # y_data = y_data.to(device)
        output = model(x_data)
        output = torch.max(output.data,dim=1)[1]
        Total=len(output)
        output=output.cpu()


        ACC=metrics.accuracy_score(output,y_data)*100
        MAP=metrics.precision_score(output,y_data,average='macro')*100
        MAR=metrics.recall_score(output,y_data,average='macro')*100
        MAF=metrics.f1_score(output,y_data,average='macro')*100
        ACClist.append(round(ACC, 2)), MAPlist.append(round(MAP, 2)), MARlist.append(round(MAR, 2)), MAFlist.append(
            round(MAF, 2))
        # print('Total: %d,ACC: %.5f %%,MAP= %.5f %%,MAR=%.5f %%,MAF= %.5f %%' % (Total,ACC,MAP,MAR,MAF))
def modeltest(model,x_data,y_data,begin,end):
    train(model,num_epochs, np.delete(x_data,np.arange(begin,end+1),axis=0), np.delete(y_data ,np.arange(begin,end+1)))
    if is_trainsettest:
        t_data(model, np.delete(x_data,np.arange(begin,end+1),axis=0), np.delete(y_data ,np.arange(begin,end+1)))
    t_data(model,x_data[begin:end+1], y_data[begin:end+1])
def outprint():
    if is_trainsettest==False:
        print('ACC:',ACClist)
        print('MAP:',MAPlist)
        print('MAR:',MARlist)
        print('MAF:',MAFlist)
        print('Average ACC: %.2f'% (sum(ACClist)/foldnum))
        print('Average MAP: %.2f' % (sum(MAPlist) / foldnum))
        print('Average MAR: %.2f' % (sum(MARlist) / foldnum))
        print('Average MAF: %.2f'% (sum(MAFlist)/foldnum))
    else:
        print('Train ACC:',ACClist[0:foldnum*2:2])
        print('Test ACC:',ACClist[1:foldnum*2:2])
        print('Train MAP:',MAPlist[0:foldnum*2:2])
        print('Test MAP:',MAPlist[1:foldnum*2:2])
        print('Train MAR:',MARlist[0:foldnum*2:2])
        print('Test MAR:',MARlist[1:foldnum*2:2])
        print('Train MAF:',MAFlist[0:foldnum*2:2])
        print('Test MAF:',MAFlist[1:foldnum*2:2])
        print('Average ACC: %.2f'% (sum(ACClist[1:foldnum*2:2])/foldnum))
        print('Average MAP: %.2f' % (sum(MAPlist[1:foldnum*2:2]) / foldnum))
        print('Average MAR: %.2f' % (sum(MARlist[1:foldnum*2:2]) / foldnum))
        print('Average MAF: %.2f'% (sum(MAFlist[1:foldnum*2:2])/foldnum))


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('../figure/emodb_dnn')
# 更新混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def finaltest(model,x_data, y_data):
    dataset = CSVDataset(x_data, y_data)
    x_data,y_data=dataset.x_data,dataset.y_data
    with torch.no_grad():
        x_data = x_data.to(device)
        # y_data = y_data.to(device)
        output = model(x_data)
        output = torch.max(output.data,dim=1)[1]
        Total=len(output)
        output=output.cpu()
        conf_matrix = torch.zeros(7, 7)
        conf_matrix = confusion_matrix(output, y_data, conf_matrix)
        plot_confusion_matrix(conf_matrix, classes=classes, normalize=False, title='Normalized confusion matrix')
        # print('Total: %d,ACC: %.5f %%,MAP= %.5f %%,MAR=%.5f %%,MAF= %.5f %%' % (Total,ACC,MAP,MAR,MAF))

if __name__=='__main__':
    x_data , y_data = CSVDataopen(data_path)
    people=[
        [0,48],
        [49,106],
        [107,149],
        [150,187],
        [188,242],
        [243,277],
        [278,338],
        [339,407],
        [408,463],
        [464,534]
    ]
    maxacc=0
    for i in range(len(people)):
        print('第',i+1,'次验证')
        model = Net().to(device)
        modeltest(model,x_data,y_data,people[i][0],people[i][1])
        if ACClist[i]>maxacc:
            maxacc=ACClist[i]
            model1=model
    finaltest(model1,x_data,y_data)
    outprint()
 # print('AVERAGE:ACC: %f %%,MAP: %f %%,MAR: %f %%,MAF: %f %%' % (aveacc/foldnum,avemap/foldnum,avemar/foldnum,avemaf/foldnum))
    torch.save(model, './dnnmodel_emodb.pkl')
    print('Finish!')
#nohup python dnnemodb.py> dnnemodb.out &torch.nn.Module
