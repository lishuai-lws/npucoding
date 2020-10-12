# -*- coding: utf-8 -*-
import os
import wave
import numpy as np

CutTimeDef = 3  # 以1s截断文件


# CutFrameNum =0

def SetFileName(WavFileName):
    for i in range(len(files)):
        FileName = files[i]
        print("SetFileName File Name is ", FileName)
        FileName = WavFileName;






def CutFile(audio_path,i):
    files = os.listdir(audio_path)
    for file in files:
        if file[-4:] == '.wav':
            print("CutFile File Name is ", file)
            f = wave.open(audio_path + '/' + file, "rb")
            params = f.getparams()
            print(params)
            nchannels, sampwidth, framerate, nframes = params[:4]
            CutFrameNum = framerate * CutTimeDef
            # 读取格式信息
            # 一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte    单位）, 采
            # 样频率, 采样点数, 压缩类型, 压缩类型的描述。wave模块只支持非压缩的数据，因此可以忽略最后两个信息
            print("CutFrameNum=%d" % (CutFrameNum))
            print("nchannels=%d" % (nchannels))
            print("sampwidth=%d" % (sampwidth))
            print("framerate=%d" % (framerate))
            print("nframes=%d" % (nframes))
            str_data = f.readframes(nframes)  # 将波形数据转换成数组
            f.close()
            # Cutnum =nframes/framerate/CutTimeDef
            # 需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
            wave_data = np.fromstring(str_data, dtype=np.short)
            # wave_data.shape = -1, 2
            wave_data = wave_data.T
            temp_data = wave_data.T
            # StepNum = int(nframes/200)
            tmp = nframes
            pmt = temp_data
            while CutFrameNum >= nframes:
                temp_data = np.concatenate((temp_data, pmt), axis=0)
                nframes = nframes + tmp
            StepNum = CutFrameNum
            StepTotalNum = 0;
            haha = 0
            while StepTotalNum < nframes:
                print("Stemp=%d" % (haha))
                File = '../iemocap/segment/Session' +str(i+1)+'/'+ file[0:-4]
                isExists = os.path.exists(File)
                if not isExists:
                    os.makedirs(File)
                newname = '{:0>2d}.wav'.format(haha + 1)
                Name = file[0:-4] + "-" + newname
                FileName = File + "/" + Name
                print(FileName)
                if haha == 0:
                    temp_dataTemp = temp_data[StepNum * (haha): StepNum * (haha + 1) + 1]
                    StepTotalNum = StepNum * (haha + 1)
                    temp = StepNum * (haha + 1)
                elif (nframes > (temp + (StepNum / 2))):
                    temp_dataTemp = temp_data[int(temp - StepNum / 2): int(temp + StepNum / 2) + 1]
                    StepTotalNum = int(temp + StepNum / 2)
                    temp = int(temp + StepNum / 2)
                else:
                    temp_dataTemp = temp_data[nframes - StepNum: nframes + 1]
                    StepTotalNum = nframes
                haha = haha + 1;
                temp_dataTemp.shape = 1, -1
                temp_dataTemp = temp_dataTemp.astype(np.short)  # 打开WAV文档
                f = wave.open(FileName, "wb")  #
                # 配置声道数、量化位数和取样频率
                f.setnchannels(nchannels)
                f.setsampwidth(sampwidth)
                f.setframerate(framerate)
                # 将wav_data转换为二进制数据写入文件
                f.writeframes(temp_dataTemp.tostring())
                f.close()


if __name__ == '__main__':
    for i in range(5):
        audio_path = '../iemocap/Session'+str(i+1)
        CutFile(audio_path,i)
    print("Run Over")
