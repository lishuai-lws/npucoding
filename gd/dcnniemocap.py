import torchvision
import torchvision.transforms as transforms
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import KFold


batch_size=64
learning_rate = 0.0001
epochnum = 100
foldnum=1
is_trainsettest=True
Dropoutrate=0.2
ACClist,MAPlist,MARlist,MAFlist=[],[],[],[]
tlosslist,vlosslist=[],[]
classes=['angry','happy','neutral','sad']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def labeltoint(emotion):
    if emotion=='ang':
        return 0
    elif emotion=='hap':
        return 1
    elif emotion=='neu':
        return 2
    else:
        return 3

class Dataset(Dataset):
    def __init__(self,image, label):
        data_path = '../iemocap/spectrogram'
        images=[]
        labels=[]
        for i in range(len(image)):
            imagepath=data_path+'/'+image[i]
            imagelist=os.listdir(imagepath)
            for j in imagelist:
                images.append(os.path.join(imagepath,j))
                labels.append(labeltoint(label[i]))
        self.images=np.array(images)
        self.labels=torch.LongTensor(np.array(labels))
        self.transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (255, 255, 255)),
            # transforms.Normalize((55.395330,169149787,121.918715),(31.461523,28.240152,20.652902)) #标准化RGB
        ]
        )
    def __getitem__(self, item):
        image_path,label=self.images[item],self.labels[item]
        pil_image=Image.open(image_path).convert('RGB')
        data=self.transforms(pil_image)
        return data,label
    def __len__(self):
        return len(self.images)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5,stride=2),  # in_channels=3,out_channels=32,kernel_size=3,stride=2
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(Dropoutrate),
            # 32*63
            nn.Conv2d(8, 16, kernel_size=5,stride=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(Dropoutrate),
            # 64*15
            nn.Conv2d(16, 32, kernel_size=2,stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(Dropoutrate),
        )
        self.classer=nn.Sequential(
            nn.Linear(1728,4),
            # nn.ReLU(),
            # nn.Linear(128,7)
        )


    def forward(self,x):
        x=self.conv(x)
        # print(x.size())
        x=x.view(x.size()[0],-1)
        # print(x.size())
        x=self.classer(x)
        return x

def train(model,images,labels,epochnum,imagestest,labelstest):
    trainset = Dataset(images, labels)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epochnum=epochnum
    trainloader=DataLoader(dataset=trainset,
                           batch_size=batch_size,
                           shuffle=True
                           )
    for epoch in range(epochnum):
        # print('Epoch [{}/{}]'.format(epoch+1,epochnum))
        runing_loss=0
        for i,(inputs,labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            runing_loss+=loss.item()
            loss.backward()
            optimizer.step()
        tlosslist.append(runing_loss/(len(trainset)//batch_size+1))
        # print('Epoch [{}/{}] loss:{}'.format(epoch + 1, epochnum,runing_loss/(len(trainset)//batch_size+1)))
        # 计算测试集的损失值
        testset = Dataset(imagestest, labelstest)
        test_loader = DataLoader(dataset=testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=2
                                 )
        testloss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                testloss += loss.item()
        vlosslist.append(testloss / (len(testset) // batch_size + 1))
        print('Epoch [{}/{}] loss:{:.4f} Testloss:{:.4f}'.format(epoch + 1, \
                                                                 epochnum,runing_loss/(len(trainset)//batch_size+1),testloss / (len(testset) // batch_size + 1)))

def modeltest(model,images,labels):
    testset=Dataset(images,labels)
    # images,labels=testset.images,testset.labels
    test_loader = DataLoader(dataset=testset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=2
                             )
    Total=0
    ACC=0
    MAP=0
    MAR=0
    MAF=0
    with torch.no_grad():
        for images,labels in test_loader:
            images = images.to(device)
            # labels = labels.to(device)
            testnum=labels.size(0)
            Total+=testnum
            output = model(images)
            output = torch.max(output.data,dim=1)[1]
            output=output.cpu()
            ACC+=metrics.accuracy_score(output,labels,normalize=False)
            MAP+=metrics.precision_score(output,labels,average='macro')*testnum
            MAR+=metrics.recall_score(output,labels,average='macro')*testnum
            MAF+=metrics.f1_score(output,labels,average='macro')*testnum
        ACClist.append(round(100*ACC/Total,2)),MAPlist.append(round(100*MAP/Total,2)),MARlist.append(round(100*MAR/Total,2)),MAFlist.append(round(100*MAF/Total,2))
        # print('Total: %d ACC: %.5f %%,MAP= %.5f %%,MAR=%.5f %%,MAF= %.5f %%' % (Total,100*ACC/Total,100*MAP/Total,100*MAR/Total,100*MAF/Total))
        # return ACC/Total, MAP/Total, MAR/Total, MAF/Total

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
    # for data in traindatalist:
    #     data = data.strip('\n')
    #     traindata.append(data.split())
    for data in testdatalist:
        data=data.strip('\n')
        testdata.append(data.split())
    return np.array(traindata),np.array(testdata)

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
    plt.savefig('../figure/iemocap_dcnntest')
# 更新混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def finaltest(model,traindata,testdata):
    conf_matrix = torch.zeros(4, 4)
    testset=Dataset(traindata[:,0],traindata[:,1])
    # images,labels=testset.images,testset.labels
    test_loader = DataLoader(dataset=testset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=2
                             )
    with torch.no_grad():
        for images,labels in test_loader:
            images = images.to(device)
            # labels = labels.to(device)
            testnum=labels.size(0)
            output = model(images)
            output = torch.max(output.data,dim=1)[1]
            output=output.cpu()

            conf_matrix = confusion_matrix(output, labels, conf_matrix)

    testset = Dataset(testdata[:,0],testdata[:,1])
    # images,labels=testset.images,testset.labels
    test_loader = DataLoader(dataset=testset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=2
                             )
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            # labels = labels.to(device)
            testnum = labels.size(0)
            output = model(images)
            output = torch.max(output.data, dim=1)[1]
            output = output.cpu()
            conf_matrix = confusion_matrix(output, labels, conf_matrix)

        plot_confusion_matrix(conf_matrix, classes=classes, normalize=False, title='Normalized confusion matrix')
        # print('Total: %d ACC: %.5f %%,MAP= %.5f %%,MAR=%.5f %%,MAF= %.5f %%' % (Total,100*ACC/Total,100*MAP/Total,100*MAR/Total,100*MAF/Total))
        # return ACC/Total, MAP/Total, MAR/Total, MAF/Total

if __name__=='__main__':
    sessionpath='../iemocap/'
    maxacc=0
    for i in range(foldnum):
        print('第',i+1,'次验证')
        traindata,testdata=getfold(i,sessionpath)
        model = Net().to(device)
        train(model,traindata[:,0],traindata[:,1],epochnum,testdata[:,0],testdata[:,1])
        if is_trainsettest:
            modeltest(model,traindata[:,0],traindata[:,1])
        modeltest(model,testdata[:,0],testdata[:,1])
        if ACClist[i]>maxacc:
            maxacc=ACClist[i]
            model1=model
    # traindata, testdata = getfold(1, sessionpath)
    # finaltest(model1,traindata,testdata)
    outprint()
    plt.figure()
    plt.plot(tlosslist, label='train')
    plt.plot(vlosslist, label='test')
    plt.legend()
    plt.savefig('../figure/iemocaplossfigure100')
    # torch.save(model1,'./dcnnmodel_iemocap.pkl')
    print('Finish!')

# nohup python dcnniemocap.py > dcnniemocap.out &

# 43
