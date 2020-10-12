import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from sklearn import metrics
from torchvision import transforms
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import KFold
from sklearn import preprocessing

loss_list = []
batch_size = 24
epochnum = 150
Dropoutrate=0.3
learning_rate = 0.001
ACClist,MAPlist,MARlist,MAFlist=[],[],[],[]
foldnum=5
is_trainsettest=True
classes=['angry','happy','neutral','sad']
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def labeltoint(emotion):
    if emotion=='ang':
        return 0
    elif emotion=='hap':
        return 1
    elif emotion=='neu':
        return 2
    else:
        return 3
class CSVDataset(Dataset):
    def __init__(self,x_data,y_data):
        self.len = x_data.shape[0]
        self.x_data = torch.Tensor(x_data)
        self.y_data = torch.LongTensor(y_data)
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
            torch.nn.Softmax(),

            torch.nn.Linear(1024, 256),
            torch.nn.Dropout(Dropoutrate),
            torch.nn.BatchNorm1d(256),
            torch.nn.Softmax(),

            torch.nn.Linear(256, 128),
            torch.nn.Dropout(Dropoutrate),
            torch.nn.BatchNorm1d(128),
            torch.nn.Softmax(),

            torch.nn.Linear(128, 64),
            torch.nn.Dropout(Dropoutrate),
            torch.nn.BatchNorm1d(64),
            torch.nn.Softmax(),

            torch.nn.Linear(64, 7),
        )
    def forward(self, x):
        x=self.linear(x)
        return x

def train(model,num_epochs,train_dataset):
    # train_dataset = CSVDataset(x_data, y_data)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2
                              )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        batch_num = 0
        running_loss = 0
        for i, data in enumerate(train_loader):
            x_data,y_data = data
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            output = model(x_data)
            # y_data=y_data.float()
            loss = criterion(output, y_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            batch_num=batch_num+1
            # print('Epoch [%d/%d], Batch[%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size + 1, loss.item()))
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, running_loss/(len(train_dataset) // batch_size + 1)))
        loss_list.append(running_loss)

def modeltest(model,dataset):
    # dataset = CSVDataset(x_data, y_data)
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
    # print('Total: %d,ACC: %.2f %%,MAP= %.2f %%,MAR=%.2f %%,MAF= %.2f %%' % (Total,ACC,MAP,MAR,MAF))

def getfold(i,foldspath):
    list=[0,1,2,3,4]
    list.remove(i)
    traindata=[]
    testdata=[]
    for j in list:
        trainsessionpath = foldspath   + 'Session' + str(j + 1) + '.txt'
        traindatalist = open(trainsessionpath)
        for data in traindatalist:
            data = data.strip('\n')
            traindata.append(data.split())
    testsessionpath = foldspath + 'Session' + str(i + 1) + '.txt'
    testdatalist = open(testsessionpath)
    for data in testdatalist:
        data=data.strip('\n')
        testdata.append(data.split())
    traindata , testdata = np.array(traindata) , np.array(testdata)
    x_data=np.concatenate((traindata[:,0],testdata[:,0]),axis=0)
    y_data = np.concatenate((traindata[:,1],testdata[:,1]))
    # print(x_data)
    datapath = '../iemocap/functional/'
    xdata = []
    ydata = []
    for i in range(len(x_data)):
        funpath = datapath + x_data[i] + '.csv'
        now_data = np.loadtxt(funpath, delimiter=",", skiprows=1, dtype='float32', unpack=False).tolist()
        now_data = now_data[:][1:6903]
        xdata.append(now_data)
        ydata.append(labeltoint(y_data[i]))
    xdata = preprocessing.scale(xdata)
    traindataset= CSVDataset(np.array(xdata)[:traindata.shape[0]],np.array(ydata)[:traindata.shape[0]])
    testdataset = CSVDataset(np.array(xdata)[traindata.shape[0]:],np.array(ydata)[traindata.shape[0]:])
    return traindataset, testdataset

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

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,):
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
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
    # # x,y轴长度一致(问题1解决办法）
    # plt.axis("equal")
    # # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    # ax = plt.gca()  # 获得当前axis
    # left, right = plt.xlim()  # 获得x轴最大最小值
    # ax.spines['left'].set_position(('data', left))
    # ax.spines['right'].set_position(('data', right))
    # for edge_i in ['top', 'bottom', 'right', 'left']:
    #     ax.spines[edge_i].set_edgecolor("white")
    # # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

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
    plt.savefig('../figure/iemocap_dnn')
# 更新混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def finaltest(model,dataset1,dataset2):
    with torch.no_grad():
        x_data, y_data = dataset1.x_data, dataset1.y_data
        x_data = x_data.to(device)
        # y_data = y_data.to(device)
        output = model(x_data)
        output = torch.max(output.data,dim=1)[1]
        output=output.cpu()
        conf_matrix = torch.zeros(4, 4)
        conf_matrix = confusion_matrix(output, y_data, conf_matrix)

        x_data, y_data = dataset2.x_data, dataset2.y_data
        x_data = x_data.to(device)
        # y_data = y_data.to(device)
        output = model(x_data)
        output = torch.max(output.data,dim=1)[1]
        output=output.cpu()
        conf_matrix = confusion_matrix(output, y_data, conf_matrix)
        plot_confusion_matrix(conf_matrix, classes=classes, normalize=False, title='Normalized confusion matrix',)
if __name__=='__main__':
    sessionpath='../iemocap/'
    maxacc=0
    for i in range(foldnum):
        print('第',i+1,'次验证')
        traindataset, testdataset = getfold(i, sessionpath)
        model = Net().to(device)
        train(model, epochnum, traindataset)
        if is_trainsettest:
            modeltest(model,traindataset)
        modeltest(model, testdataset)
        if ACClist[i]>maxacc:
            maxacc=ACClist[i]
            model1=model
    traindataset, testdataset = getfold(1, sessionpath)
    finaltest(model1, traindataset,testdataset )
    outprint()

    # print('AVERAGE:ACC: %f %%,MAP: %f %%,MAR: %f %%,MAF: %f %%' % (aveacc/foldnum,avemap/foldnum,avemar/foldnum,avemaf/foldnum))
    torch.save(model, './dnnmodel_iemocap.pkl')
    print('Finish!')
# nohup python sdnniemocap.py > dnniemocap.out &
# 59.85
